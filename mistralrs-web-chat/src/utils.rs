use std::path::PathBuf;

/// Determine the base cache directory for the application.
/// Uses XDG_CACHE_HOME or falls back to ~/.cache/mistralrs-web-chat.
pub fn get_cache_dir() -> PathBuf {
    // XDG_CACHE_HOME or default to ~/.cache
    let cache_home = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from(".cache"))
        });
    cache_home.join("mistralrs-web-chat")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;

    // Mutex to ensure tests that modify the environment don't race
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn set_env_var(key: &str, value: Option<&str>) -> Option<std::ffi::OsString> {
        let old_value = env::var_os(key);
        if let Some(v) = value {
            env::set_var(key, v);
        } else {
            env::remove_var(key);
        }
        old_value
    }

    fn restore_env_var(key: &str, old_value: Option<std::ffi::OsString>) {
        if let Some(v) = old_value {
            env::set_var(key, v);
        } else {
            env::remove_var(key);
        }
    }

    #[test]
    fn test_get_cache_dir_xdg_cache_home() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let old_xdg = set_env_var("XDG_CACHE_HOME", Some("/custom/xdg/cache"));
        let old_home = set_env_var("HOME", Some("/home/user"));
        let dir = get_cache_dir();
        assert_eq!(dir, PathBuf::from("/custom/xdg/cache/mistralrs-web-chat"));
        restore_env_var("HOME", old_home);
        restore_env_var("XDG_CACHE_HOME", old_xdg);
    }

    #[test]
    fn test_get_cache_dir_home_fallback() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let old_xdg = set_env_var("XDG_CACHE_HOME", None);
        let old_home = set_env_var("HOME", Some("/home/user"));
        let dir = get_cache_dir();
        assert_eq!(dir, PathBuf::from("/home/user/.cache/mistralrs-web-chat"));
        restore_env_var("HOME", old_home);
        restore_env_var("XDG_CACHE_HOME", old_xdg);
    }

    #[test]
    fn test_get_cache_dir_no_env_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let old_xdg = set_env_var("XDG_CACHE_HOME", None);
        let old_home = set_env_var("HOME", None);
        let dir = get_cache_dir();
        assert_eq!(dir, PathBuf::from(".cache/mistralrs-web-chat"));
        restore_env_var("HOME", old_home);
        restore_env_var("XDG_CACHE_HOME", old_xdg);
    }
}
