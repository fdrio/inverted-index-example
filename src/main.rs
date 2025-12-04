use std::net::SocketAddr;
use std::sync::Arc;

use axum::{Json, Router, extract::State, routing};
use log::{Level, log};
use serde::{Deserialize, Serialize};
use tokio::{self, sync::Mutex};

mod index;

use index::inverted_index::InvertedIndex;

#[derive(Clone)]
struct AppState {
    index: Arc<Mutex<InvertedIndex>>,
}

#[derive(Deserialize)]
pub struct IndexRequest {
    pub doc_id: usize,
    pub text: String,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub query: String,
    pub hits: Vec<usize>,
}

// Index a document into the inverted index
async fn index(State(state): State<AppState>, Json(req): Json<IndexRequest>) -> Json<SearchResult> {
    let tokens: Vec<(String, usize)> = req
        .text
        .split_whitespace()
        .enumerate()
        .map(|(pos, term)| (term.to_lowercase(), pos))
        .collect();

    {
        let mut idx = state.index.lock().await;
        idx.add_document(req.doc_id, tokens);
    }

    // For indexing we just return an empty hit list
    Json(SearchResult {
        query: String::from("indexed"),
        hits: Vec::new(),
    })
}

// Search the inverted index for a single term
async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResult> {
    let hits = {
        let idx = state.index.lock().await;
        idx.rank(&req.query).unwrap_or_default()
    };

    Json(SearchResult {
        query: req.query,
        hits,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let state = AppState {
        index: Arc::new(Mutex::new(InvertedIndex::default())),
    };

    let app = Router::new()
        .route("/index/", routing::post(index))
        .route("/search/", routing::post(search))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:3000".parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log!(Level::Info, "Starting server {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
