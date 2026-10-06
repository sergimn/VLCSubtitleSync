-- luacheck configuration for the VLC Lua scripts (VLC 3 embeds Lua 5.1/5.2/LuaJIT).
std = "lua51+lua52"
max_line_length = 120
exclude_files = { ".cache/**", ".venv/**" }

-- Provided by VLC to every script type.
read_globals = { "vlc" }

files["src/vlcsubsync/lua/intf/subsync.lua"] = {
    -- test hook: the script exports itself when SUBSYNC_TEST is set
    globals = { "subsync" },
    read_globals = { "SUBSYNC_TEST" },
}

files["src/vlcsubsync/lua/extensions/subsync_ext.lua"] = {
    -- extension entry points looked up by VLC as globals
    globals = {
        "descriptor", "activate", "deactivate", "close", "menu", "trigger_menu",
        "input_changed", "meta_changed", "playing_changed", "subsync_ext",
    },
    read_globals = { "SUBSYNC_TEST" },
}

files["tests/lua/**"] = {
    globals = { "vlc" },
}
files["tests/lua/vlc_mock.lua"].unused_args = false
