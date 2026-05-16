use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use serde_json::json;
use std::sync::Arc;
use tokio::fs;
use tracing::error;
use uuid::Uuid;

use mistralrs::speech_utils;
use std::fs::File;
use std::path::PathBuf;

use crate::models::LoadedModel;
use crate::types::{
    AppState,
    ChatFile,
    DeleteChatRequest,
    LoadChatRequest,
    NewChatRequest,
    RenameChatRequest,
    SelectRequest,
    // Append partial assistant messages
    // (defined below)
};
use crate::utils::get_cache_dir;
use serde::Deserialize;

fn validate_image_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    // Check MIME type first
    if let Some(mime) = content_type {
        if !mime.starts_with("image/") {
            return Err("File must be an image");
        }
    }

    // Validate file extension
    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "svg" => Ok(ext),
        "" => Err("No file extension"),
        _ => Err("Unsupported image format"),
    }
}

fn validate_audio_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    // Check MIME type first
    if let Some(mime) = content_type {
        if !mime.starts_with("audio/") {
            return Err("File must be an audio file");
        }
    }

    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac" | "opus" | "webm" => Ok(ext),
        "" => Err("No file extension"),
        _ => Err("Unsupported audio format"),
    }
}

/// Accepts multipart audio upload, stores under `cache/uploads/`, returns its URL.
pub async fn upload_audio(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => {
            let orig_filename = field.file_name().map(|s| s.to_string());
            let content_type_opt = field.content_type().map(|s| s.to_string());

            let ext = match validate_audio_upload(
                orig_filename.as_deref(),
                content_type_opt.as_deref(),
            ) {
                Ok(ext) => ext,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };

            // Read bytes (limit 50MB like images)
            let data = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("multipart bytes error: {}", e);
                    let err = format!("{e} {e:?}").to_lowercase();
                    let msg = if err.contains("exceed")
                        || err.contains("limit")
                        || err.contains("length")
                    {
                        "audio too large (limit 50 MB)"
                    } else {
                        "failed to read upload"
                    };
                    return (StatusCode::BAD_REQUEST, msg).into_response();
                }
            };

            const MAX_SIZE: usize = 50 * 1024 * 1024;
            if data.len() > MAX_SIZE {
                return (StatusCode::BAD_REQUEST, "audio too large (limit 50 MB)").into_response();
            }

            // Ensure upload directory exists
            let uploads_dir = get_cache_dir().join("uploads");
            if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
                error!("create uploads dir error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create uploads directory",
                )
                    .into_response();
            }

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let filepath = uploads_dir.join(&filename);
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                error!("write upload error: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save audio").into_response();
            }

            let url = filepath.to_string_lossy();
            (StatusCode::OK, Json(json!({ "url": url }))).into_response()
        }
        Ok(None) => (StatusCode::BAD_REQUEST, "missing audio part").into_response(),
        Err(e) => {
            error!("multipart field error: {}", e);
            let err = format!("{e} {e:?}").to_lowercase();
            let msg = if err.contains("exceed") || err.contains("limit") || err.contains("length") {
                "audio too large (limit 50 MB)"
            } else {
                "failed to read upload"
            };
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}

fn validate_text_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    // Check MIME type first (allow text/* and application/json)
    if let Some(mime) = content_type {
        if !mime.starts_with("text/")
            && mime != "application/json"
            && mime != "application/javascript"
        {
            // Also allow common binary MIME types that are actually text
            if !matches!(
                mime,
                "application/octet-stream"
                    | "application/x-python"
                    | "application/x-rust"
                    | "application/x-sh"
            ) {
                return Err("File must be a text file");
            }
        }
    }

    // Validate file extension
    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        // Text files
        "txt" | "md" | "markdown" | "log" | "csv" | "tsv" | "json" | "xml" | "yaml" | "yml"
        | "toml" | "ini" | "cfg" | "conf" => Ok(ext),
        // Code files
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "html" | "htm" | "css" | "scss" | "sass"
        | "less" => Ok(ext),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" | "java" | "kt" | "swift" | "go"
        | "rb" | "php" => Ok(ext),
        // GPU and shader languages
        "cu" | "cuh" | "cl" | "ptx" | "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc"
        | "tese" | "hlsl" | "metal" | "wgsl" => Ok(ext),
        // Shell and other scripts
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" | "sql" | "dockerfile"
        | "makefile" => Ok(ext),
        "r" | "R" | "scala" | "clj" | "cljs" | "hs" | "elm" | "ex" | "exs" | "erl" | "fs"
        | "fsx" | "ml" | "mli" => Ok(ext),
        "vue" | "svelte" | "astro" | "lua" | "nim" | "zig" | "d" | "dart" | "jl" | "pl" | "pm"
        | "tcl" => Ok(ext),
        // Config and other text-like files
        "gitignore" | "dockerignore" | "editorconfig" | "env" | "htaccess" => Ok(ext),
        "" => {
            // Files without extension - check the filename
            if let Some(name) = filename {
                let name_lower = name.to_lowercase();
                if matches!(
                    name_lower.as_str(),
                    "readme"
                        | "license"
                        | "changelog"
                        | "makefile"
                        | "dockerfile"
                        | "vagrantfile"
                        | "gemfile"
                        | "rakefile"
                ) {
                    return Ok("txt".to_string());
                }
            }
            Err("No file extension")
        }
        _ => Err("Unsupported text file format"),
    }
}

/// Accepts multipart image upload, stores it under `cache/uploads/`, and returns its URL.
pub async fn upload_image(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Expect single "image" part
    if let Ok(Some(field)) = multipart.next_field().await {
        // Clone filename and content type to avoid borrowing `field`
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let ext = match validate_image_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(extension) => extension,
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "image too large (limit 50 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        const MAX_SIZE: usize = 50 * 1024 * 1024; // 50MB
        if data.len() > MAX_SIZE {
            return (StatusCode::BAD_REQUEST, "image too large (limit 50 MB)").into_response();
        }

        if image::load_from_memory(&data).is_err() {
            return (StatusCode::BAD_REQUEST, "invalid image file").into_response();
        }

        // Determine upload directory under cache
        let base_cache = get_cache_dir();
        let upload_dir = base_cache.join("uploads");
        if let Err(e) = tokio::fs::create_dir_all(&upload_dir).await {
            error!("mkdir error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }
        // Build unique filename and write
        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let path = upload_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&path, &data).await {
            error!("write error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }
        return (
            StatusCode::OK,
            Json(json!({ "url": path.to_string_lossy() })),
        )
            .into_response();
    }

    (StatusCode::BAD_REQUEST, "no image field").into_response()
}

/// Accepts multipart text file upload and returns the file content
pub async fn upload_text(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Expect single "file" part
    if let Ok(Some(field)) = multipart.next_field().await {
        // Clone filename and content type to avoid borrowing `field`
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let _ext = match validate_text_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(extension) => extension,
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "file too large (limit 10 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        const MAX_SIZE: usize = 10 * 1024 * 1024; // 10MB for text files
        if data.len() > MAX_SIZE {
            return (StatusCode::BAD_REQUEST, "file too large (limit 10 MB)").into_response();
        }

        // Try to decode as UTF-8
        let content = match String::from_utf8(data.to_vec()) {
            Ok(text) => text,
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "file is not valid UTF-8 text").into_response();
            }
        };

        // Limit content length for safety
        const MAX_CHARS: usize = 1_000_000; // 1 million characters
        if content.len() > MAX_CHARS {
            return (
                StatusCode::BAD_REQUEST,
                "file content too large (limit 1M characters)",
            )
                .into_response();
        }

        return (
            StatusCode::OK,
            Json(json!({
                "content": content,
                "filename": orig_filename.unwrap_or_else(|| "untitled".to_string()),
                "size": data.len()
            })),
        )
            .into_response();
    }

    (StatusCode::BAD_REQUEST, "no file field").into_response()
}

pub async fn list_models(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let items: Vec<_> = app
        .models
        .iter()
        .map(|(n, m)| {
            let kind = match m {
                LoadedModel::Text(_) => "text",
                LoadedModel::Multimodal(_) => "multimodal",
                LoadedModel::Speech(_) => "speech",
            };
            json!({ "name": n, "kind": kind })
        })
        .collect();
    Json(json!({ "models": items }))
}

pub async fn select_model(
    State(app): State<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> impl IntoResponse {
    if let Some(model_loaded) = app.models.get(&req.name) {
        {
            let mut cur = app.current.write().await;
            *cur = Some(req.name.clone());
        }
        // --- sync the active chat file so future loads use the correct model ---
        if let Some(chat_id) = app.current_chat.read().await.clone() {
            let path = format!("{}/{}.json", app.chats_dir, chat_id);
            if let Ok(data) = fs::read(&path).await {
                if let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&data) {
                    chat.model = req.name.clone();
                    chat.kind = match model_loaded {
                        LoadedModel::Text(_) => "text".into(),
                        LoadedModel::Multimodal(_) => "multimodal".into(),
                        LoadedModel::Speech(_) => "speech".into(),
                    };
                    // ignore write errors; not fatal for select_model
                    if let Ok(bytes) = serde_json::to_vec_pretty(&chat) {
                        let _ = tokio::fs::write(&path, bytes).await;
                    }
                }
            }
        }
        (StatusCode::OK, "Model selected").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Model not found").into_response()
    }
}

pub async fn list_chats(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let mut chats = Vec::new();
    if let Ok(mut dir) = fs::read_dir(&app.chats_dir).await {
        while let Ok(Some(entry)) = dir.next_entry().await {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") {
                    let id = name.trim_end_matches(".json");
                    let data = fs::read(format!("{}/{}", app.chats_dir, name)).await.ok();
                    let (title, created) = data
                        .and_then(|bytes| serde_json::from_slice::<ChatFile>(&bytes).ok())
                        .map(|c| (c.title, c.created_at))
                        .map(|(title, created)| (title.unwrap_or_default(), created))
                        .unwrap_or_else(|| (String::new(), String::new()));
                    chats.push(json!({ "id": id, "title": title, "created_at": created }));
                }
            }
        }
    }
    Json(json!({ "chats": chats }))
}

pub async fn new_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<NewChatRequest>,
) -> impl IntoResponse {
    let mut id_guard = app.next_chat_id.write().await;
    let id = *id_guard;
    *id_guard += 1;
    drop(id_guard);

    let chat_id = format!("chat_{id}");
    let path = format!("{}/{}.json", app.chats_dir, chat_id);

    let kind = if let Some(m) = app.models.get(&req.model) {
        match m {
            LoadedModel::Text(_) => "text",
            LoadedModel::Multimodal(_) => "multimodal",
            LoadedModel::Speech(_) => "speech",
        }
    } else {
        "text"
    }
    .to_string();

    let chat = ChatFile {
        title: None,
        model: req.model.clone(),
        kind,
        created_at: Utc::now().to_rfc3339(),
        messages: Vec::new(),
    };
    let _ = fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap()).await;

    {
        let mut cur_chat = app.current_chat.write().await;
        *cur_chat = Some(chat_id.clone());
        let mut cur_model = app.current.write().await;
        *cur_model = Some(req.model.clone());
    }

    Json(json!({ "id": chat_id }))
}

pub async fn delete_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<DeleteChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Err(e) = tokio::fs::remove_file(&path).await {
        error!("delete chat error: {}", e);
        return (StatusCode::NOT_FOUND, "chat not found").into_response();
    }
    {
        let mut cur_chat = app.current_chat.write().await;
        if cur_chat.as_ref() == Some(&req.id) {
            *cur_chat = None;
            let mut cur_model = app.current.write().await;
            *cur_model = None;
        }
    }
    (StatusCode::OK, "Deleted").into_response()
}

pub async fn load_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<LoadChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    match fs::read(&path).await {
        Ok(data) => match serde_json::from_slice::<ChatFile>(&data) {
            Ok(chat) => {
                {
                    let mut cur_chat = app.current_chat.write().await;
                    *cur_chat = Some(req.id.clone());
                    if app.models.contains_key(&chat.model) {
                        let mut cur_model = app.current.write().await;
                        *cur_model = Some(chat.model.clone());
                    }
                }
                Json(json!({
                    "id": req.id,
                    "title": chat.title.clone().unwrap_or_default(),
                    "model": chat.model,
                    "kind": chat.kind,
                    "created_at": chat.created_at.clone(),
                    "messages": chat.messages
                }))
                .into_response()
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "corrupt chat").into_response(),
        },
        Err(_) => (StatusCode::NOT_FOUND, "chat not found").into_response(),
    }
}

pub async fn rename_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<RenameChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Ok(data) = fs::read(&path).await {
        if let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&data) {
            chat.title = Some(req.title.clone());
            if let Ok(bytes) = serde_json::to_vec_pretty(&chat) {
                let _ = tokio::fs::write(&path, bytes).await;
                return (StatusCode::OK, "Renamed").into_response();
            }
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "rename failed").into_response()
}

/// Request to append a (partial) assistant message to a chat
#[derive(Deserialize)]
pub struct AppendMessageRequest {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

/// Appends a partial assistant response (or any role) to the chat file.
pub async fn append_message(
    State(app): State<Arc<AppState>>,
    Json(req): Json<AppendMessageRequest>,
) -> impl IntoResponse {
    if let Err(e) =
        crate::chat::append_chat_message(&app, &req.id, &req.role, &req.content, req.images).await
    {
        error!("append message error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "append failed").into_response();
    }
    (StatusCode::OK, "Appended").into_response()
}
/// Request to generate speech from text
#[derive(Deserialize)]
pub struct GenerateSpeechRequest {
    pub text: String,
}

/// Get current settings (default generation params and search status)
pub async fn get_settings(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "defaults": {
            "temperature": app.default_params.temperature,
            "top_p": app.default_params.top_p,
            "top_k": app.default_params.top_k,
            "max_tokens": app.default_params.max_tokens,
            "repetition_penalty": app.default_params.repetition_penalty,
            "system_prompt": app.default_params.system_prompt,
        },
        "search_enabled": app.search_enabled,
    }))
}

/// Endpoint to generate speech (.wav) for a given prompt using a speech model
pub async fn generate_speech(
    State(app): State<Arc<AppState>>,
    Json(req): Json<GenerateSpeechRequest>,
) -> impl IntoResponse {
    // Determine selected model
    let model_name = {
        let cur = app.current.read().await;
        if let Some(name) = &*cur {
            name.clone()
        } else {
            return (StatusCode::BAD_REQUEST, "No model selected").into_response();
        }
    };
    // Ensure model exists and is a speech model
    if let Some(LoadedModel::Speech(m)) = app.models.get(&model_name) {
        // Generate speech
        let (pcm, rate, channels) = match m.generate_speech(req.text).await {
            Ok(res) => res,
            Err(e) => {
                error!("speech generation error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "speech generation failed",
                )
                    .into_response();
            }
        };
        // Write WAV file
        let filename = format!("{}.wav", Uuid::new_v4());
        let filepath = PathBuf::from(&app.speech_dir).join(&filename);
        if let Err(e) = File::create(&filepath).and_then(|mut f| {
            speech_utils::write_pcm_as_wav(&mut f, &pcm, rate as u32, channels as u16)
        }) {
            error!("failed to write wav file: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to write wav file",
            )
                .into_response();
        }
        // Return URL for client download
        let url = format!("/speech/{filename}");
        (StatusCode::OK, Json(json!({ "url": url }))).into_response()
    } else {
        (
            StatusCode::BAD_REQUEST,
            "Selected model is not a speech model",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        extract::DefaultBodyLimit,
        http::{header, Method, Request, StatusCode},
        routing::post,
        Router,
    };
    use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
    use indexmap::IndexMap;
    use std::{
        env,
        ffi::OsString,
        path::{Path, PathBuf},
        sync::Arc,
    };
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::{
        types::{AppState, GenerationParams},
        utils::ENV_MUTEX,
    };

    struct EnvVarGuard {
        key: &'static str,
        old_value: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&Path>) -> Self {
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

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            models: IndexMap::new(),
            current: RwLock::new(None),
            chats_dir: "target/test-web-chat/chats".to_string(),
            speech_dir: "target/test-web-chat/speech".to_string(),
            current_chat: RwLock::new(None),
            next_chat_id: RwLock::new(1),
            default_params: GenerationParams::default(),
            search_enabled: false,
        })
    }

    fn app_for_uploads() -> Router {
        Router::new()
            .route("/api/upload_image", post(upload_image))
            .route("/api/upload_audio", post(upload_audio))
            .route("/api/upload_text", post(upload_text))
            .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
            .with_state(test_state())
    }

    fn multipart_body(
        boundary: &str,
        field_name: &str,
        filename: Option<&str>,
        content_type: Option<&str>,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{field_name}\"").as_bytes(),
        );
        if let Some(filename) = filename {
            body.extend_from_slice(format!("; filename=\"{filename}\"").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        if let Some(content_type) = content_type {
            body.extend_from_slice(format!("Content-Type: {content_type}\r\n").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(payload);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        body
    }

    async fn post_multipart(
        app: Router,
        path: &str,
        boundary: &str,
        body: Vec<u8>,
    ) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header(
                        header::CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    fn tiny_png() -> Vec<u8> {
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(&[255, 0, 0, 255], 1, 1, ColorType::Rgba8.into())
            .unwrap();
        bytes
    }

    #[test]
    fn validate_image_upload_accepts_happy_paths() {
        assert_eq!(
            validate_image_upload(Some("photo.jpg"), Some("image/jpeg")),
            Ok("jpg".to_string())
        );
        assert_eq!(
            validate_image_upload(Some("diagram.PNG"), None),
            Ok("png".to_string())
        );
    }

    #[test]
    fn validate_image_upload_rejects_error_paths() {
        assert_eq!(
            validate_image_upload(Some("photo.jpg"), Some("text/plain")),
            Err("File must be an image")
        );
        assert_eq!(
            validate_image_upload(None, Some("image/png")),
            Err("No filename provided")
        );
        assert_eq!(
            validate_image_upload(Some("photo."), Some("image/png")),
            Err("No file extension")
        );
        assert_eq!(
            validate_image_upload(Some("photo.txt"), Some("image/png")),
            Err("Unsupported image format")
        );
    }

    #[test]
    fn validate_audio_upload_accepts_happy_paths() {
        assert_eq!(
            validate_audio_upload(Some("clip.wav"), Some("audio/wav")),
            Ok("wav".to_string())
        );
        assert_eq!(
            validate_audio_upload(Some("clip.MP3"), None),
            Ok("mp3".to_string())
        );
    }

    #[test]
    fn validate_audio_upload_rejects_error_paths() {
        assert_eq!(
            validate_audio_upload(Some("clip.wav"), Some("video/mp4")),
            Err("File must be an audio file")
        );
        assert_eq!(
            validate_audio_upload(None, Some("audio/wav")),
            Err("No filename provided")
        );
        assert_eq!(
            validate_audio_upload(Some("clip."), Some("audio/wav")),
            Err("No file extension")
        );
        assert_eq!(
            validate_audio_upload(Some("clip.exe"), Some("audio/wav")),
            Err("Unsupported audio format")
        );
    }

    #[test]
    fn validate_text_upload_accepts_happy_paths() {
        assert_eq!(
            validate_text_upload(Some("notes.txt"), Some("text/plain")),
            Ok("txt".to_string())
        );
        assert_eq!(
            validate_text_upload(Some("main.rs"), Some("application/x-rust")),
            Ok("rs".to_string())
        );
        assert_eq!(
            validate_text_upload(Some("Dockerfile"), None),
            Ok("dockerfile".to_string())
        );
    }

    #[test]
    fn validate_text_upload_rejects_error_paths() {
        assert_eq!(
            validate_text_upload(None, Some("text/plain")),
            Err("No filename provided")
        );
        assert_eq!(
            validate_text_upload(Some("notes."), Some("text/plain")),
            Err("No file extension")
        );
        assert_eq!(
            validate_text_upload(Some("notes.exe"), Some("text/plain")),
            Err("Unsupported text file format")
        );
        assert_eq!(
            validate_text_upload(Some("notes.txt"), Some("image/png")),
            Err("File must be a text file")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn upload_image_accepts_valid_png() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|err| err.into_inner());
        let cache_root = PathBuf::from(format!(
            "{}/mistralrs-web-chat-image-ok-{}",
            env::temp_dir().display(),
            Uuid::new_v4()
        ));
        let _xdg = EnvVarGuard::set("XDG_CACHE_HOME", Some(&cache_root));
        let body = multipart_body(
            "boundary",
            "image",
            Some("tiny.png"),
            Some("image/png"),
            &tiny_png(),
        );

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_image", "boundary", body).await;

        assert_eq!(status, StatusCode::OK);
        assert!(response_body.contains("\"url\""));
        let _ = tokio::fs::remove_dir_all(cache_root).await;
    }

    #[tokio::test]
    async fn upload_image_rejects_malformed_multipart() {
        let response = app_for_uploads()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/upload_image")
                    .header(
                        header::CONTENT_TYPE,
                        "multipart/form-data; boundary=boundary",
                    )
                    .body(Body::from("this is not multipart"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upload_image_rejects_missing_part() {
        let (status, response_body) = post_multipart(
            app_for_uploads(),
            "/api/upload_image",
            "boundary",
            b"--boundary--\r\n".to_vec(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "no image field");
    }

    #[tokio::test]
    async fn upload_image_rejects_invalid_image_file() {
        let body = multipart_body(
            "boundary",
            "image",
            Some("fake.jpg"),
            Some("image/jpeg"),
            b"not an image",
        );

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_image", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "invalid image file");
    }

    #[tokio::test]
    async fn upload_image_rejects_payload_over_50mb() {
        let app = Router::new()
            .route("/api/upload_image", post(upload_image))
            .layer(DefaultBodyLimit::disable())
            .with_state(test_state());
        let payload = vec![0; 50 * 1024 * 1024 + 1];
        let body = multipart_body(
            "boundary",
            "image",
            Some("large.png"),
            Some("image/png"),
            &payload,
        );

        let (status, response_body) =
            post_multipart(app, "/api/upload_image", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "image too large (limit 50 MB)");
    }

    #[tokio::test]
    async fn upload_audio_rejects_missing_part() {
        let (status, response_body) = post_multipart(
            app_for_uploads(),
            "/api/upload_audio",
            "boundary",
            b"--boundary--\r\n".to_vec(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "missing audio part");
    }

    #[tokio::test]
    async fn upload_audio_rejects_malformed_multipart() {
        let response = app_for_uploads()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/upload_audio")
                    .header(
                        header::CONTENT_TYPE,
                        "multipart/form-data; boundary=boundary",
                    )
                    .body(Body::from("this is not multipart"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upload_audio_rejects_invalid_type() {
        let body = multipart_body(
            "boundary",
            "audio",
            Some("clip.wav"),
            Some("text/plain"),
            b"audio data",
        );

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_audio", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "File must be an audio file");
    }

    #[tokio::test]
    async fn upload_audio_rejects_missing_filename() {
        let body = multipart_body("boundary", "audio", None, Some("audio/wav"), b"audio data");

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_audio", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "No filename provided");
    }

    #[tokio::test]
    async fn upload_audio_rejects_unsupported_format() {
        let body = multipart_body(
            "boundary",
            "audio",
            Some("clip.exe"),
            Some("audio/wav"),
            b"audio data",
        );

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_audio", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "Unsupported audio format");
    }

    #[tokio::test]
    async fn upload_audio_rejects_payload_over_limit() {
        let app = Router::new()
            .route("/api/upload_audio", post(upload_audio))
            .layer(DefaultBodyLimit::disable())
            .with_state(test_state());
        let payload = vec![0; 50 * 1024 * 1024 + 1];
        let body = multipart_body(
            "boundary",
            "audio",
            Some("clip.wav"),
            Some("audio/wav"),
            &payload,
        );

        let (status, response_body) =
            post_multipart(app, "/api/upload_audio", "boundary", body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response_body, "audio too large (limit 50 MB)");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn upload_audio_reports_directory_creation_failure() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|err| err.into_inner());
        let blocking_file =
            env::temp_dir().join(format!("mistralrs-web-chat-cache-file-{}", Uuid::new_v4()));
        tokio::fs::write(&blocking_file, b"not a directory")
            .await
            .unwrap();
        let _xdg = EnvVarGuard::set("XDG_CACHE_HOME", Some(&blocking_file));
        let body = multipart_body(
            "boundary",
            "audio",
            Some("clip.wav"),
            Some("audio/wav"),
            b"audio data",
        );

        let (status, response_body) =
            post_multipart(app_for_uploads(), "/api/upload_audio", "boundary", body).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response_body, "failed to create uploads directory");
        let _ = tokio::fs::remove_file(blocking_file).await;
    }
}
