"""exe name -> app category. Pure (standard library only): the app sampler and ml/ training share it."""

CATEGORIES = ("ide", "browser", "terminal", "game", "chat", "other")

# exe stem (lowercase, no ".exe") → category. shortcut: hand-made list from BEHACOM + common dev tools;
# DevDesk will let the user edit it in settings (plan: app_category mapping). Not here on purpose: java/javaw (also
# Minecraft and other Java apps), and ApplicationFrameHost (every UWP app shows up as it; "other" until the probe
# resolves the hosted app).
_EXE_CATEGORY = {
    **dict.fromkeys(
        ["code", "devenv", "idea64", "pycharm64", "eclipse", "codeblocks", "sublime_text", "notepad++",
         "gedit", "texniccenter", "gummi", "wxmaxima", "sqlitebrowser", "gns3",
         "androidemulator", "wireshark"], "ide"),
    **dict.fromkeys(
        ["chrome", "firefox", "msedge", "opera", "brave", "chromium-browser", "acwebbrowser", "whale"], "browser"),
    **dict.fromkeys(
        ["gnome-terminal-server", "tilix", "konsole", "cmd", "powershell", "pwsh", "windowsterminal",
         "putty", "wezterm-gui", "openconsole"], "terminal"),
    **dict.fromkeys(
        ["csgo", "league of legends", "leagueclientux", "leagueclient", "steam", "winamax poker",
         "pokerstars", "pokerstarsbr", "tslgame", "epicgameslauncher", "lostark"], "game"),
    **dict.fromkeys(
        ["skype", "skypeforlinux", "telegram", "whatsapp", "thunderbird", "ts3client_win64", "slack",
         "kakaotalk", "discord"], "chat"),
}


def exe_stem(exe_name) -> str:
    """One name per app: lowercase, no arguments, no ".exe" ("Firefox.exe" = "firefox" on Linux)."""
    stem = str(exe_name).strip().lower()
    stem = stem.split(" --")[0]  # "chromium-browser --enable-pinch" → "chromium-browser"
    if stem.endswith(".exe"):
        stem = stem[:-4]
    return stem


def categorize(exe_name) -> str:
    return _EXE_CATEGORY.get(exe_stem(exe_name), "other")
