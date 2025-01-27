//! ## Task Description
//!
//! The goal is to develop a backend service for shortening URLs using CQRS
//! (Command Query Responsibility Segregation) and ES (Event Sourcing)
//! approaches. The service should support the following features:
//!
//! ## Functional Requirements
//!
//! ### Creating a short link with a random slug
//!
//! The user sends a long URL, and the service returns a shortened URL with a
//! random slug.
//!
//! ### Creating a short link with a predefined slug
//!
//! The user sends a long URL along with a predefined slug, and the service
//! checks if the slug is unique. If it is unique, the service creates the short
//! link.
//!
//! ### Counting the number of redirects for the link
//!
//! - Every time a user accesses the short link, the click count should
//!   increment.
//! - The click count can be retrieved via an API.
//!
//! ### CQRS+ES Architecture
//!
//! CQRS: Commands (creating links, updating click count) are separated from
//! queries (retrieving link information).
//!
//! Event Sourcing: All state changes (link creation, click count update) must be
//! recorded as events, which can be replayed to reconstruct the system's state.
//!
//! ### Technical Requirements
//!
//! - The service must be built using CQRS and Event Sourcing approaches.
//! - The service must be possible to run in Rust Playground (so no database like
//!   Postgres is allowed)
//! - Public API already written for this task must not be changed (any change to
//!   the public API items must be considered as breaking change).
//! - Event Sourcing should be actively utilized for implementing logic, rather
//!   than existing without a clear purpose.

#![allow(unused_variables, dead_code)]

/// All possible errors of the [`UrlShortenerService`].
#[derive(Debug, PartialEq)]
pub enum ShortenerError {
    /// This error occurs when an invalid [`Url`] is provided for shortening.
    InvalidUrl,

    /// This error occurs when an attempt is made to use a slug (custom alias)
    /// that already exists.
    SlugAlreadyInUse,

    /// This error occurs when the provided [`Slug`] does not map to any existing
    /// short link.
    SlugNotFound,
}

use serde::{ Serialize, Deserialize };

/// A unique string (or alias) that represents the shortened version of the
/// URL.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Slug(pub String);

/// The original URL that the short link points to.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Url(pub String);

/// Shortened URL representation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ShortLink {
    /// A unique string (or alias) that represents the shortened version of the
    /// URL.
    pub slug: Slug,

    /// The original URL that the short link points to.
    pub url: Url,
}

/// Statistics of the [`ShortLink`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stats {
    /// [`ShortLink`] to which this [`Stats`] are related.
    pub link: ShortLink,

    /// Count of redirects of the [`ShortLink`].
    pub redirects: u64,
}

/// Commands for CQRS.
pub mod commands {
    use super::{ShortLink, ShortenerError, Slug, Url};

    /// Trait for command handlers.
    pub trait CommandHandler {
        /// Creates a new short link. It accepts the original url and an
        /// optional [`Slug`]. If a [`Slug`] is not provided, the service will generate
        /// one. Returns the newly created [`ShortLink`].
        ///
        /// ## Errors
        ///
        /// See [`ShortenerError`].
        fn handle_create_short_link(
            &mut self,
            url: Url,
            slug: Option<Slug>,
        ) -> Result<ShortLink, ShortenerError>;

        /// Processes a redirection by [`Slug`], returning the associated
        /// [`ShortLink`] or a [`ShortenerError`].
        fn handle_redirect(
            &mut self,
            slug: Slug,
        ) -> Result<ShortLink, ShortenerError>;
    }
}

/// Queries for CQRS
pub mod queries {
    use super::{ShortenerError, Slug, Stats};

    /// Trait for query handlers.
    pub trait QueryHandler {
        /// Returns the [`Stats`] for a specific [`ShortLink`], such as the
        /// number of redirects (clicks).
        ///
        /// [`ShortLink`]: super::ShortLink
        fn get_stats(&self, slug: Slug) -> Result<Stats, ShortenerError>;
    }
}

/// CQRS and Event Sourcing-based service implementation
#[derive(Clone, Copy)]
pub struct UrlShortenerService {
}

impl UrlShortenerService {
    /// Creates a new instance of the service
    ///
    pub fn new() -> Self {
        Self {}
    }

    pub fn get_slugs(&self, skip: u32, take: u32) -> Vec<String> {
        Vec::new()
    }
}

impl commands::CommandHandler for UrlShortenerService {
    fn handle_create_short_link(
        &mut self,
        url: Url,
        slug: Option<Slug>,
    ) -> Result<ShortLink, ShortenerError> {
        todo!("Implement the logic for creating a short link")
    }

    fn handle_redirect(
        &mut self,
        slug: Slug,
    ) -> Result<ShortLink, ShortenerError> {
        todo!("Implement the logic for redirection and incrementing the click counter")
    }
}

impl queries::QueryHandler for UrlShortenerService {
    fn get_stats(&self, slug: Slug) -> Result<Stats, ShortenerError> {
        todo!("Implement the logic for retrieving link statistics")
    }
}

use actix_web::{ web, get, post, http::header, App, HttpServer, HttpResponse, Responder };
use std::{ env, sync::{ Mutex } };

#[derive(Deserialize)]
struct CreateSlugRequest {
    url: String,
    slug: Option<String>,
}

#[post("/api/slugs")]
async fn create_slug(
    service: web::Data<Mutex<UrlShortenerService>>
    , request: web::Json<CreateSlugRequest>
) -> impl Responder {
    let service = service.into_inner();
    let mut guard = service.lock().unwrap();
    let service: &mut UrlShortenerService = &mut guard;
    let request = request.into_inner();
    let url = Url(request.url);
    let slug = request.slug.map(Slug);
    match commands::CommandHandler::handle_create_short_link(service, url, slug) {
        Ok(link) => HttpResponse::Ok().json(link.slug.0),
        _ => HttpResponse::BadRequest().finish(),
    }
}

#[derive(Deserialize)]
struct GetSlugsRequest {
  skip: Option<u32>,
  take: Option<u32>,
}

#[get("/api/slugs")]
async fn get_slugs(
    service: web::Data<Mutex<UrlShortenerService>>
    , request: web::Query<GetSlugsRequest>
) -> HttpResponse {
    let request = request.into_inner();
    let mutex = service.into_inner();
    let ref service: UrlShortenerService = *mutex.lock().unwrap();
    let slugs = service.get_slugs(
        request.skip.unwrap_or(0),
        request.take.unwrap_or(10));
    HttpResponse::Ok().json(slugs)
}

#[get("/api/slugs/{slug}")]
async fn get_slug(service: web::Data<Mutex<UrlShortenerService>>, slug: web::Path<String>) -> impl Responder {
    let slug = Slug(slug.into_inner());
    let mutex = service.into_inner();
    let ref service: UrlShortenerService = *mutex.lock().unwrap();
    match queries::QueryHandler::get_stats(service, slug) {
        Ok(stats) => HttpResponse::Ok().json(stats),
        _ => HttpResponse::NotFound().finish(),
    }
}

#[get("/{slug}")]
async fn visit_slug(service: web::Data<Mutex<UrlShortenerService>>, slug: web::Path<String>) -> HttpResponse {
    let service = service.into_inner();
    let mut guard = service.lock().unwrap();
    let service: &mut UrlShortenerService = &mut guard;
    let slug = Slug(slug.into_inner());
    match commands::CommandHandler::handle_redirect(service, slug) {
        Ok(link) => HttpResponse::TemporaryRedirect()
            .insert_header((header::LOCATION, link.url.0))
            .finish(),
        // only relevant error is `SlugNotFound`
        _ => HttpResponse::NotFound().finish(),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let service_url = env::var("SHURL").expect("SHURL service url must be provided");
    let state = web::Data::new(Mutex::new(UrlShortenerService::new()));
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::clone(&state))
            .service(create_slug)
            .service(get_slugs)
            .service(get_slug)
            .service(visit_slug)
    })
    .bind(service_url)?
    .run()
    .await
}
