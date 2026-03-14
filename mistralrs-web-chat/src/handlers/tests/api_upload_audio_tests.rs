use std::sync::Arc;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    routing::post,
    Router,
};
use tower::ServiceExt; // for `oneshot` and `ready`

use crate::handlers::api::upload_audio;
use crate::types::AppState;
use indexmap::IndexMap;

#[tokio::test]
async fn test_upload_audio_missing_part() {
    let app_state = Arc::new(AppState {
        models: IndexMap::new(),
        current: tokio::sync::RwLock::new(None),
        chats_dir: ".".to_string(),
        speech_dir: ".".to_string(),
        current_chat: tokio::sync::RwLock::new(None),
        next_chat_id: tokio::sync::RwLock::new(1),
        default_params: Default::default(),
        search_enabled: false,
    });

    let app = Router::new()
        .route("/api/upload_audio", post(upload_audio))
        .with_state(app_state);

    // Create a multipart request with no parts
    let boundary = "------------------------14737809831466499882746641449";
    let body = format!("--{}--\r\n", boundary);

    let request = Request::builder()
        .method("POST")
        .uri("/api/upload_audio")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
