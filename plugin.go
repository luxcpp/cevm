package cevm

import (
	"os"
	"path/filepath"
)

// PluginPath returns the absolute path to the cevm VM plugin binary.
// Used by lux CLI and universe Makefiles to locate the built plugin.
func PluginPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "work", "luxcpp", "evm", "build", "bin", "cevm")
}

// PluginExists reports whether the cevm plugin binary is present on disk.
func PluginExists() bool {
	p := PluginPath()
	if p == "" {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// VMID returns the VM ID for the cevm plugin.
// This is the identifier used by Lux subnet configuration to reference
// this VM in the plugin directory.
func VMID() string {
	return "cevm"
}
