// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file main.cpp
/// cevm: C++ EVM Lux VM plugin binary.
///
/// Speaks ZAP protocol natively to luxd. Launch modes:
///   - LUX_VM_TRANSPORT=zap: perform ZAP handshake and serve VM requests
///   - No env var: print version and exit (diagnostic mode)

#include "evm_backend.hpp"
#include "zap_server.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{

/// ZAP protocol version (must match version.RPCChainVMProtocol in luxd).
constexpr uint32_t PROTOCOL_VERSION = 42;

/// Perform ZAP handshake with luxd runtime engine.
///
/// 1. Read LUX_VM_RUNTIME_ENGINE_ADDR from env
/// 2. Bind TCP listener on 127.0.0.1:0
/// 3. Connect to runtime engine, send handshake
/// 4. Read 1-byte ACK
/// 5. Accept inbound connection from luxd
/// 6. Return the connected fd
int zap_handshake()
{
    const char* engine_addr = std::getenv("LUX_VM_RUNTIME_ENGINE_ADDR");
    if (engine_addr == nullptr)
    {
        fprintf(stderr, "[cevm] error: LUX_VM_RUNTIME_ENGINE_ADDR not set\n");
        return -1;
    }

    // Parse engine address (host:port).
    std::string addr_str(engine_addr);
    auto colon = addr_str.rfind(':');
    if (colon == std::string::npos)
    {
        fprintf(stderr, "[cevm] error: invalid engine address: %s\n", engine_addr);
        return -1;
    }
    std::string host = addr_str.substr(0, colon);
    int port = std::atoi(addr_str.substr(colon + 1).c_str());

    // Bind listener on random port.
    int listener_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener_fd < 0)
    {
        perror("[cevm] socket");
        return -1;
    }

    int opt = 1;
    ::setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind_addr.sin_port = 0;  // Random port.

    if (::bind(listener_fd, reinterpret_cast<struct sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0)
    {
        perror("[cevm] bind");
        ::close(listener_fd);
        return -1;
    }

    if (::listen(listener_fd, 1) < 0)
    {
        perror("[cevm] listen");
        ::close(listener_fd);
        return -1;
    }

    // Get the actual bound port.
    struct sockaddr_in actual_addr{};
    socklen_t addr_len = sizeof(actual_addr);
    ::getsockname(listener_fd, reinterpret_cast<struct sockaddr*>(&actual_addr), &addr_len);
    int vm_port = ntohs(actual_addr.sin_port);

    char vm_addr_str[64];
    std::snprintf(vm_addr_str, sizeof(vm_addr_str), "127.0.0.1:%d", vm_port);
    fprintf(stderr, "[cevm] ZAP listener bound on %s\n", vm_addr_str);

    // Connect to runtime engine.
    int engine_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (engine_fd < 0)
    {
        perror("[cevm] socket (engine)");
        ::close(listener_fd);
        return -1;
    }

    struct sockaddr_in engine_sockaddr{};
    engine_sockaddr.sin_family = AF_INET;
    engine_sockaddr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, host.c_str(), &engine_sockaddr.sin_addr);

    if (::connect(engine_fd, reinterpret_cast<struct sockaddr*>(&engine_sockaddr),
                  sizeof(engine_sockaddr)) < 0)
    {
        perror("[cevm] connect to runtime engine");
        ::close(engine_fd);
        ::close(listener_fd);
        return -1;
    }

    // Build handshake: [4-byte BE total_len][4-byte BE protocol_version][vm_addr_bytes]
    size_t addr_bytes_len = std::strlen(vm_addr_str);
    uint32_t total_len = static_cast<uint32_t>(4 + addr_bytes_len);

    uint8_t handshake_buf[256];
    size_t pos = 0;

    // 4-byte BE total_len
    handshake_buf[pos++] = static_cast<uint8_t>(total_len >> 24);
    handshake_buf[pos++] = static_cast<uint8_t>(total_len >> 16);
    handshake_buf[pos++] = static_cast<uint8_t>(total_len >> 8);
    handshake_buf[pos++] = static_cast<uint8_t>(total_len);

    // 4-byte BE protocol version
    handshake_buf[pos++] = static_cast<uint8_t>(PROTOCOL_VERSION >> 24);
    handshake_buf[pos++] = static_cast<uint8_t>(PROTOCOL_VERSION >> 16);
    handshake_buf[pos++] = static_cast<uint8_t>(PROTOCOL_VERSION >> 8);
    handshake_buf[pos++] = static_cast<uint8_t>(PROTOCOL_VERSION);

    // VM address string
    std::memcpy(handshake_buf + pos, vm_addr_str, addr_bytes_len);
    pos += addr_bytes_len;

    if (!cevm::write_exact(engine_fd, handshake_buf, pos))
    {
        fprintf(stderr, "[cevm] error: failed to send handshake\n");
        ::close(engine_fd);
        ::close(listener_fd);
        return -1;
    }

    // Read 1-byte ACK.
    uint8_t ack = 0;
    if (!cevm::read_exact(engine_fd, &ack, 1) || ack != 0x01)
    {
        fprintf(stderr, "[cevm] error: handshake rejected (ack=%u)\n", ack);
        ::close(engine_fd);
        ::close(listener_fd);
        return -1;
    }

    // Close runtime connection.
    ::close(engine_fd);

    // Accept inbound connection from luxd.
    fprintf(stderr, "[cevm] waiting for luxd to connect back...\n");
    struct sockaddr_in peer_addr{};
    socklen_t peer_len = sizeof(peer_addr);
    int client_fd = ::accept(listener_fd, reinterpret_cast<struct sockaddr*>(&peer_addr), &peer_len);
    ::close(listener_fd);

    if (client_fd < 0)
    {
        perror("[cevm] accept");
        return -1;
    }

    char peer_str[64];
    inet_ntop(AF_INET, &peer_addr.sin_addr, peer_str, sizeof(peer_str));
    fprintf(stderr, "[cevm] luxd connected from %s:%d\n", peer_str, ntohs(peer_addr.sin_port));

    return client_fd;
}

}  // namespace

int main(int argc, char* argv[])
{
    // Check for version flag.
    if (argc > 1 && (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0))
    {
        std::printf("cevm/0.1.0\n");
        return 0;
    }

    // Check transport mode.
    const char* transport = std::getenv("LUX_VM_TRANSPORT");
    if (transport == nullptr || std::strcmp(transport, "zap") != 0)
    {
        std::printf("cevm/0.1.0 - C++ EVM Lux VM plugin\n");
        std::printf("usage: set LUX_VM_TRANSPORT=zap and LUX_VM_RUNTIME_ENGINE_ADDR to run as VM plugin\n");
        return 0;
    }

    fprintf(stderr, "[cevm] starting in ZAP mode\n");

    // Perform ZAP handshake.
    int fd = zap_handshake();
    if (fd < 0)
    {
        fprintf(stderr, "[cevm] error: cannot connect to runtime engine\n");
        return 1;
    }

    // Create backend and serve.
    cevm::EvmBackend backend;
    cevm::ZapServer::serve(fd, backend);

    ::close(fd);
    fprintf(stderr, "[cevm] exiting\n");
    return 0;
}
