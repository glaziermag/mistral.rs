use axum::{
    body::Body,
    extract::Request,
    http::{header, Method, StatusCode},
    routing::post,
    Router,
};
use std::sync::Arc;
use tower::ServiceExt;

use crate::handlers::api::upload_image;
use crate::types::AppState;

#[tokio::test]
async fn test_upload_image_too_large() {
    let app_state = Arc::new(AppState {
        models: Default::default(),
        current: tokio::sync::RwLock::new(None),
        chats_dir: "".to_string(),
        speech_dir: "".to_string(),
        current_chat: tokio::sync::RwLock::new(None),
        next_chat_id: tokio::sync::RwLock::new(0),
        default_params: Default::default(),
        search_enabled: false,
    });

    // Use a router without any DefaultBodyLimit layer, so that we can test
    // the internal `if data.len() > MAX_SIZE` block (50MB)
    let app = Router::new()
        .route("/api/upload_image", post(upload_image))
        // Since `DefaultBodyLimit` is 2MB by default in axum, we need to disable it
        // to reach our 50MB internal handler block check!
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(app_state);

    let boundary = "------------------------Boundary";

    // Create a body larger than 50MB
    // We'll create exactly 50 * 1024 * 1024 + 1 bytes of payload
    let payload = vec![b'A'; 50 * 1024 * 1024 + 10];

    let mut body_data = Vec::new();
    body_data.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body_data.extend_from_slice(b"Content-Disposition: form-data; name=\"image\"; filename=\"large.png\"\r\n");
    body_data.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_data.extend_from_slice(&payload);
    body_data.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/upload_image")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_data))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_str, "image too large (limit 50 MB)");
}
