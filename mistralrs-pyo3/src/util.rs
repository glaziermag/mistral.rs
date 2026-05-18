use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use image::DynamicImage;
use mistralrs_core::AudioInput;
use mistralrs_core::ResponseErr;
use pyo3::{exceptions::PyValueError, PyErr};

pub(crate) struct PyApiErr(pub(crate) PyErr);
pub(crate) type PyApiResult<T> = Result<T, PyApiErr>;

impl std::fmt::Debug for PyApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for PyApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for PyApiErr {}

impl From<reqwest::Error> for PyApiErr {
    fn from(value: reqwest::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<std::io::Error> for PyApiErr {
    fn from(value: std::io::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<anyhow::Error> for PyApiErr {
    fn from(value: anyhow::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<&candle_core::Error> for PyApiErr {
    fn from(value: &candle_core::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<serde_json::Error> for PyApiErr {
    fn from(value: serde_json::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<mistralrs_core::MistralRsError> for PyApiErr {
    fn from(value: mistralrs_core::MistralRsError) -> Self {
        Self::from(value.to_string())
    }
}

impl From<String> for PyApiErr {
    fn from(value: String) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

impl From<&str> for PyApiErr {
    fn from(value: &str) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

impl From<PyApiErr> for PyErr {
    fn from(value: PyApiErr) -> Self {
        value.0
    }
}

impl From<Box<ResponseErr>> for PyApiErr {
    fn from(value: Box<ResponseErr>) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

fn resolve_local_media_path(path: &Path, url: &url::Url) -> PyApiResult<PathBuf> {
    let cwd = std::env::current_dir()
        .and_then(|path| path.canonicalize())
        .map_err(|e| format!("Could not resolve current directory: {e}"))?;
    let resolved = path
        .canonicalize()
        .map_err(|e| format!("Could not resolve local file path: {url}: {e}"))?;

    if !resolved.starts_with(&cwd) {
        return Err(PyApiErr::from(format!(
            "Access denied: Local file path is outside the current working directory: {url}"
        )));
    }

    Ok(resolved)
}

pub(crate) fn parse_image_url(url_unparsed: &str) -> PyApiResult<DynamicImage> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| format!("Could not parse file path: {url_unparsed}"))?
    } else {
        url::Url::parse(url_unparsed).map_err(|_| {
            format!(
                "Invalid source '{}': not a valid URL (http/https/data) and file not found. \
                 Use a full URL, a data URL, or a file path that exists.",
                url_unparsed
            )
        })?
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        // Read from http
        match reqwest::blocking::get(url.clone()) {
            Ok(http_resp) => http_resp.bytes()?.to_vec(),
            Err(e) => return Err(PyApiErr::from(format!("{e}"))),
        }
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| format!("Could not parse file path: {url}"))?;

        let path = resolve_local_media_path(&path, &url)?;

        if let Ok(mut f) = File::open(&path) {
            // Read from local file
            let metadata = fs::metadata(&path)?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer)?;
            buffer
        } else {
            return Err(PyApiErr::from(format!(
                "Could not open file at path: {url}"
            )));
        }
    } else if url.scheme() == "data" {
        // Decode with base64
        let data_url = data_url::DataUrl::process(url.as_str()).map_err(|e| format!("{e}"))?;
        data_url.decode_to_vec().map_err(|e| format!("{e}"))?.0
    } else {
        return Err(PyApiErr::from(format!(
            "Unsupported URL scheme: {}",
            url.scheme()
        )));
    };

    image::load_from_memory(&bytes).map_err(|e| PyApiErr::from(format!("{e}")))
}

/// Parses and loads an audio file from a URL, file path, or data URL.
/// Mirrors `parse_image_url` but returns an `AudioInput`.
pub(crate) fn parse_audio_url(url_unparsed: &str) -> PyApiResult<AudioInput> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| format!("Could not parse file path: {url_unparsed}"))?
    } else {
        url::Url::parse(url_unparsed).map_err(|_| {
            format!(
                "Invalid source '{}': not a valid URL (http/https/data) and file not found. \
                 Use a full URL, a data URL, or a file path that exists.",
                url_unparsed
            )
        })?
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        match reqwest::blocking::get(url.clone()) {
            Ok(http_resp) => http_resp
                .bytes()
                .map_err(|e| PyApiErr::from(format!("{e}")))?
                .to_vec(),
            Err(e) => return Err(PyApiErr::from(format!("{e}"))),
        }
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| format!("Could not parse file path: {url}"))?;

        let path = resolve_local_media_path(&path, &url)?;

        if let Ok(mut f) = File::open(&path) {
            let metadata = fs::metadata(&path)?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer)?;
            buffer
        } else {
            return Err(PyApiErr::from(format!(
                "Could not open file at path: {url}"
            )));
        }
    } else if url.scheme() == "data" {
        let data_url = data_url::DataUrl::process(url.as_str()).map_err(|e| format!("{e}"))?;
        data_url.decode_to_vec().map_err(|e| format!("{e}"))?.0
    } else {
        return Err(PyApiErr::from(format!(
            "Unsupported URL scheme: {}",
            url.scheme()
        )));
    };

    AudioInput::from_bytes(&bytes).map_err(|e| PyApiErr::from(format!("{e}")))
}
#[cfg(test)]
mod local_media_path_tests {
    use super::*;
    use std::{
        env, fs,
        io::{Read, Write},
        net::TcpListener,
        path::{Path, PathBuf},
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    const PNG_1X1: &[u8] = &[
        0x42, 0x4d, 0x3a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x13, 0x0b, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00,
    ];

    fn wav_16khz_mono() -> Vec<u8> {
        let samples: [i16; 8] = [0; 8];
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    struct Fixture {
        old_cwd: PathBuf,
        root: PathBuf,
        outside: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let base = env::temp_dir().join(format!(
                "mistralrs-local-media-{name}-{}-{nonce}",
                std::process::id()
            ));
            let root = base.join("root");
            let outside = base.join("outside");
            fs::create_dir_all(&root).unwrap();
            fs::create_dir_all(&outside).unwrap();
            fs::write(root.join("inside.png"), PNG_1X1).unwrap();
            fs::write(root.join("inside.wav"), wav_16khz_mono()).unwrap();
            fs::write(outside.join("outside.png"), PNG_1X1).unwrap();
            fs::write(outside.join("outside.wav"), wav_16khz_mono()).unwrap();
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(outside.join("outside.png"), root.join("link.png"))
                    .unwrap();
                std::os::unix::fs::symlink(outside.join("outside.wav"), root.join("link.wav"))
                    .unwrap();
            }
            let old_cwd = env::current_dir().unwrap();
            env::set_current_dir(&root).unwrap();
            Self {
                old_cwd,
                root,
                outside,
            }
        }

        fn outside_image(&self) -> PathBuf {
            self.outside.join("outside.png")
        }

        fn outside_audio(&self) -> PathBuf {
            self.outside.join("outside.wav")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.old_cwd);
            let _ = fs::remove_dir_all(self.root.parent().unwrap());
        }
    }

    fn file_url(path: &Path) -> String {
        url::Url::from_file_path(path).unwrap().to_string()
    }

    fn assert_access_denied<T>(result: PyApiResult<T>) {
        assert!(result.is_err(), "expected local media path to be denied");
    }

    fn assert_access_allowed<T>(result: PyApiResult<T>) {
        assert!(result.is_ok(), "expected local media path to be allowed");
    }

    fn serve_once(bytes: Vec<u8>, content_type: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            )
            .unwrap();
            stream.write_all(&bytes).unwrap();
        });
        format!("http://{addr}/media")
    }

    #[test]
    fn image_blocks_file_url_etc_passwd() {
        let _fixture = Fixture::new("image-etc-passwd");
        assert_access_denied(parse_image_url("file:///etc/passwd"));
    }

    #[test]
    fn image_blocks_absolute_outside_cwd() {
        let fixture = Fixture::new("image-absolute");
        assert_access_denied(parse_image_url(
            &fixture.outside_image().display().to_string(),
        ));
    }

    #[test]
    fn image_blocks_traversal_escape() {
        let _fixture = Fixture::new("image-traversal");
        assert_access_denied(parse_image_url("../outside/outside.png"));
    }

    #[test]
    fn image_blocks_symlink_escape() {
        let _fixture = Fixture::new("image-symlink");
        assert_access_denied(parse_image_url("link.png"));
    }

    #[test]
    fn image_allows_inside_cwd() {
        let _fixture = Fixture::new("image-inside");
        assert_access_allowed(parse_image_url("inside.png"));
    }

    #[test]
    fn image_allows_remote_url() {
        let _fixture = Fixture::new("image-remote");
        let url = serve_once(PNG_1X1.to_vec(), "image/png");
        assert_access_allowed(parse_image_url(&url));
    }

    #[test]
    fn audio_blocks_file_url_etc_passwd() {
        let _fixture = Fixture::new("audio-etc-passwd");
        assert_access_denied(parse_audio_url("file:///etc/passwd"));
    }

    #[test]
    fn audio_blocks_absolute_outside_cwd() {
        let fixture = Fixture::new("audio-absolute");
        assert_access_denied(parse_audio_url(
            &fixture.outside_audio().display().to_string(),
        ));
    }

    #[test]
    fn audio_blocks_traversal_escape() {
        let _fixture = Fixture::new("audio-traversal");
        assert_access_denied(parse_audio_url("../outside/outside.wav"));
    }

    #[test]
    fn audio_blocks_symlink_escape() {
        let _fixture = Fixture::new("audio-symlink");
        assert_access_denied(parse_audio_url("link.wav"));
    }

    #[test]
    fn audio_allows_inside_cwd() {
        let _fixture = Fixture::new("audio-inside");
        assert_access_allowed(parse_audio_url("inside.wav"));
    }

    #[test]
    fn audio_allows_remote_url() {
        let _fixture = Fixture::new("audio-remote");
        let url = serve_once(wav_16khz_mono(), "audio/wav");
        assert_access_allowed(parse_audio_url(&url));
    }
}
