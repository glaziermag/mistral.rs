use std::path::PathBuf;

#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    use std::{env, ffi::OsString};

    struct EnvVarGuard {
        key: &'static str,
        old_value: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let old_value = env::var_os(key);
            match value {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
            Self { key, old_value }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.old_value {
                Some(value) => env::set_var(self.key, value),
                None => env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn get_cache_dir_prefers_xdg_cache_home() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|err| err.into_inner());
        let _xdg = EnvVarGuard::set("XDG_CACHE_HOME", Some("/custom/cache"));
        let _home = EnvVarGuard::set("HOME", Some("/home/example"));

        assert_eq!(
            get_cache_dir(),
            PathBuf::from("/custom/cache/mistralrs-web-chat")
        );
    }

    #[test]
    fn get_cache_dir_uses_home_fallback() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|err| err.into_inner());
        let _xdg = EnvVarGuard::set("XDG_CACHE_HOME", None);
        let _home = EnvVarGuard::set("HOME", Some("/home/example"));

        assert_eq!(
            get_cache_dir(),
            PathBuf::from("/home/example/.cache/mistralrs-web-chat")
        );
    }

    #[test]
    fn get_cache_dir_uses_relative_cache_without_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|err| err.into_inner());
        let _xdg = EnvVarGuard::set("XDG_CACHE_HOME", None);
        let _home = EnvVarGuard::set("HOME", None);

        assert_eq!(get_cache_dir(), PathBuf::from(".cache/mistralrs-web-chat"));
    }
}
