// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file zap_server.hpp
/// ZAP protocol server: TCP framing and message dispatch for Lux VM plugin.
///
/// Wire format (matches luxfi/api/zap and hanzo-zap crate):
/// - Header: 4-byte BE payload length + 1-byte message type
/// - Fields: big-endian integers, 4-byte-length-prefixed byte slices
/// - Response flag: high bit (0x80) set on reply message type byte

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <errno.h>
#include <unistd.h>

namespace cevm
{

// ---------------------------------------------------------------------------
// ZAP message types — must match luxfi/api/zap constants exactly
// ---------------------------------------------------------------------------

enum class MsgType : uint8_t
{
    Initialize = 1,
    SetState = 2,
    Shutdown = 3,
    CreateHandlers = 4,
    WaitForEvent = 6,
    Connected = 7,
    Disconnected = 8,
    BuildBlock = 9,
    ParseBlock = 10,
    GetBlock = 11,
    SetPreference = 12,
    Health = 13,
    Version = 14,
    BlockVerify = 28,
    BlockAccept = 29,
    BlockReject = 30,
};

constexpr uint8_t MSG_RESPONSE_FLAG = 0x80;
constexpr uint32_t MAX_MESSAGE_SIZE = 16 * 1024 * 1024;
constexpr size_t HEADER_SIZE = 5;  // 4-byte length + 1-byte type

// ---------------------------------------------------------------------------
// Wire helpers: big-endian reader/writer over byte buffers
// ---------------------------------------------------------------------------

class Reader
{
public:
    Reader(const uint8_t* data, size_t len) : data_{data}, len_{len} {}

    size_t remaining() const { return len_ - pos_; }

    bool read_u8(uint8_t& v)
    {
        if (remaining() < 1) return false;
        v = data_[pos_++];
        return true;
    }

    bool read_bool(bool& v)
    {
        uint8_t b;
        if (!read_u8(b)) return false;
        v = (b != 0);
        return true;
    }

    bool read_u32(uint32_t& v)
    {
        if (remaining() < 4) return false;
        v = static_cast<uint32_t>(data_[pos_]) << 24 |
            static_cast<uint32_t>(data_[pos_ + 1]) << 16 |
            static_cast<uint32_t>(data_[pos_ + 2]) << 8 |
            static_cast<uint32_t>(data_[pos_ + 3]);
        pos_ += 4;
        return true;
    }

    bool read_u64(uint64_t& v)
    {
        if (remaining() < 8) return false;
        v = 0;
        for (size_t i = 0; i < 8; ++i)
            v = (v << 8) | data_[pos_ + i];
        pos_ += 8;
        return true;
    }

    bool read_i64(int64_t& v)
    {
        uint64_t u;
        if (!read_u64(u)) return false;
        v = static_cast<int64_t>(u);
        return true;
    }

    /// Read length-prefixed bytes (4-byte BE length + data).
    bool read_bytes(std::vector<uint8_t>& out)
    {
        uint32_t len;
        if (!read_u32(len)) return false;
        if (remaining() < len) return false;
        out.assign(data_ + pos_, data_ + pos_ + len);
        pos_ += len;
        return true;
    }

    bool read_string(std::string& out)
    {
        std::vector<uint8_t> b;
        if (!read_bytes(b)) return false;
        out.assign(b.begin(), b.end());
        return true;
    }

private:
    const uint8_t* data_;
    size_t len_;
    size_t pos_ = 0;
};

class Writer
{
public:
    Writer() { buf_.reserve(4096); }

    void write_u8(uint8_t v) { buf_.push_back(v); }

    void write_bool(bool v) { write_u8(v ? 1 : 0); }

    void write_u32(uint32_t v)
    {
        buf_.push_back(static_cast<uint8_t>(v >> 24));
        buf_.push_back(static_cast<uint8_t>(v >> 16));
        buf_.push_back(static_cast<uint8_t>(v >> 8));
        buf_.push_back(static_cast<uint8_t>(v));
    }

    void write_u64(uint64_t v)
    {
        for (int i = 7; i >= 0; --i)
            buf_.push_back(static_cast<uint8_t>(v >> (i * 8)));
    }

    void write_i64(int64_t v) { write_u64(static_cast<uint64_t>(v)); }

    /// Write length-prefixed bytes.
    void write_bytes(const uint8_t* data, size_t len)
    {
        write_u32(static_cast<uint32_t>(len));
        buf_.insert(buf_.end(), data, data + len);
    }

    void write_bytes(const std::vector<uint8_t>& data)
    {
        write_bytes(data.data(), data.size());
    }

    void write_string(const std::string& s)
    {
        write_bytes(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    const uint8_t* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }
    void clear() { buf_.clear(); }

private:
    std::vector<uint8_t> buf_;
};

// ---------------------------------------------------------------------------
// POSIX socket I/O helpers
// ---------------------------------------------------------------------------

/// Read exactly n bytes from fd. Returns false on error/EOF.
inline bool read_exact(int fd, uint8_t* buf, size_t n)
{
    size_t total = 0;
    while (total < n)
    {
        auto r = ::read(fd, buf + total, n - total);
        if (r <= 0) return false;
        total += static_cast<size_t>(r);
    }
    return true;
}

/// Write exactly n bytes to fd. Returns false on error.
inline bool write_exact(int fd, const uint8_t* buf, size_t n)
{
    size_t total = 0;
    while (total < n)
    {
        auto w = ::write(fd, buf + total, n - total);
        if (w <= 0) return false;
        total += static_cast<size_t>(w);
    }
    return true;
}

// ---------------------------------------------------------------------------
// ZAP frame I/O
// ---------------------------------------------------------------------------

/// Read one ZAP frame. Returns false on disconnect.
inline bool read_frame(int fd, uint8_t& msg_type, std::vector<uint8_t>& payload)
{
    uint8_t header[HEADER_SIZE];
    if (!read_exact(fd, header, HEADER_SIZE))
        return false;

    uint32_t length = static_cast<uint32_t>(header[0]) << 24 |
                      static_cast<uint32_t>(header[1]) << 16 |
                      static_cast<uint32_t>(header[2]) << 8 |
                      static_cast<uint32_t>(header[3]);
    msg_type = header[4];

    if (length > MAX_MESSAGE_SIZE)
        return false;

    payload.resize(length);
    if (length > 0 && !read_exact(fd, payload.data(), length))
        return false;

    return true;
}

/// Write one ZAP frame.
inline bool write_frame(int fd, uint8_t msg_type, const uint8_t* payload, size_t len)
{
    uint8_t header[HEADER_SIZE];
    auto plen = static_cast<uint32_t>(len);
    header[0] = static_cast<uint8_t>(plen >> 24);
    header[1] = static_cast<uint8_t>(plen >> 16);
    header[2] = static_cast<uint8_t>(plen >> 8);
    header[3] = static_cast<uint8_t>(plen);
    header[4] = msg_type;

    if (!write_exact(fd, header, HEADER_SIZE))
        return false;
    if (len > 0 && !write_exact(fd, payload, len))
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// VmBackend interface — implemented by evm_backend.hpp
// ---------------------------------------------------------------------------

/// Abstract interface for the VM backend.
/// ZapServer dispatches decoded messages to these methods.
class VmBackend
{
public:
    virtual ~VmBackend() = default;

    /// Initialize with genesis etc. Writes response payload into w.
    virtual bool initialize(Reader& r, Writer& w) = 0;
    virtual bool set_state(Reader& r) = 0;
    virtual bool shutdown() = 0;
    virtual bool build_block(Writer& w) = 0;
    virtual bool parse_block(Reader& r, Writer& w) = 0;
    virtual bool get_block(Reader& r, Writer& w) = 0;
    virtual bool block_verify(Reader& r) = 0;
    virtual bool block_accept(Reader& r) = 0;
    virtual bool block_reject(Reader& r) = 0;
    virtual bool set_preference(Reader& r) = 0;
    virtual bool health(Writer& w) = 0;
    virtual bool version(Writer& w) = 0;
    virtual bool create_handlers(Writer& w) = 0;
    virtual bool wait_for_event(Writer& w) = 0;
};

// ---------------------------------------------------------------------------
// ZapServer — the serve loop
// ---------------------------------------------------------------------------

class ZapServer
{
public:
    /// Serve VM requests on fd until shutdown or disconnect.
    static void serve(int fd, VmBackend& backend)
    {
        bool running = true;
        while (running)
        {
            uint8_t msg_type_byte;
            std::vector<uint8_t> payload;
            if (!read_frame(fd, msg_type_byte, payload))
                break;

            // Strip response flag for dispatch.
            uint8_t raw_type = msg_type_byte & ~MSG_RESPONSE_FLAG;
            Writer resp;
            bool ok = false;

            switch (raw_type)
            {
            case static_cast<uint8_t>(MsgType::Initialize):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.initialize(r, resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::SetState):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.set_state(r);
                break;
            }
            case static_cast<uint8_t>(MsgType::Shutdown):
            {
                ok = backend.shutdown();
                running = false;
                break;
            }
            case static_cast<uint8_t>(MsgType::BuildBlock):
            {
                ok = backend.build_block(resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::ParseBlock):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.parse_block(r, resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::GetBlock):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.get_block(r, resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::SetPreference):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.set_preference(r);
                break;
            }
            case static_cast<uint8_t>(MsgType::Health):
            {
                ok = backend.health(resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::Version):
            {
                ok = backend.version(resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::CreateHandlers):
            {
                ok = backend.create_handlers(resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::WaitForEvent):
            {
                ok = backend.wait_for_event(resp);
                break;
            }
            case static_cast<uint8_t>(MsgType::BlockVerify):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.block_verify(r);
                break;
            }
            case static_cast<uint8_t>(MsgType::BlockAccept):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.block_accept(r);
                break;
            }
            case static_cast<uint8_t>(MsgType::BlockReject):
            {
                Reader r(payload.data(), payload.size());
                ok = backend.block_reject(r);
                break;
            }
            case static_cast<uint8_t>(MsgType::Connected):
            case static_cast<uint8_t>(MsgType::Disconnected):
            {
                // No-op: acknowledge peer connect/disconnect.
                ok = true;
                break;
            }
            default:
            {
                // Unknown message: send empty response.
                ok = true;
                break;
            }
            }

            // Send response with response flag set.
            uint8_t resp_type = raw_type | MSG_RESPONSE_FLAG;
            if (!ok)
            {
                // On error, send empty response (error indicated by empty payload).
                if (!write_frame(fd, resp_type, nullptr, 0))
                    break;
            }
            else
            {
                if (!write_frame(fd, resp_type, resp.data(), resp.size()))
                    break;
            }
        }
    }
};

}  // namespace cevm
