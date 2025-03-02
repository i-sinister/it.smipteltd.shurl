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
pub struct UrlShortenerService {
    cqrs: Cqrs
}

impl commands::CommandHandler for UrlShortenerService {
    fn handle_create_short_link(
        &mut self,
        url: Url,
        slug: Option<Slug>,
    ) -> Result<ShortLink, ShortenerError> {
        self.cqrs.process_new(SlugCommand::create(slug, url))
    }

    fn handle_redirect(
        &mut self,
        slug: Slug,
    ) -> Result<ShortLink, ShortenerError> {
        self.cqrs.load_and_process(&slug.0, SlugCommand::visit(slug.clone()))
    }
}

impl queries::QueryHandler for UrlShortenerService {
    fn get_stats(&self, slug: Slug) -> Result<Stats, ShortenerError> {
        self.cqrs.slug_details_query.execute(slug)
    }
}

impl UrlShortenerService {
    /// Creates a new instance of the service
    ///
    pub fn new() -> Self { Self { cqrs: Cqrs::new(None) } }

    /// Returns list of slugs
    pub fn get_slugs(&self, skip: usize, take: usize) -> Vec<String> {
        self.cqrs.slug_list_query.execute(SlugListQueryParameters { skip, take })
    }
}

struct Cqrs {
    generator: Box<dyn SlugGenerator + Sync + Send>,
    store: EventStorage,
    slug_list_query: SlugListQuery,
    slug_details_query: SlugDetailsQuery,
    used_slugs_query: UsedSlugsQuery,
}

#[derive(Clone)]
enum  SlugEvent {
    SlugCreated { slug: Slug, url: Url },
    SlugVisited { slug: Slug, redirects: u64 }
}

struct CommandContext<'a> {
    used_slugs_query: &'a UsedSlugsQuery,
    generator: &'a mut Box<dyn SlugGenerator + Sync + Send>
}

trait Aggregate {
    fn id(&self) -> String;

    /// Validates command and produces series of events
    // TODO: self should not be mutable; however in this oversiplified scenario we need it to be mutable in order to
    //       be able to assign `id` for aggregate root creation
    fn handle<'a>(&mut self, context: &'a mut CommandContext, command: SlugCommand) -> Result<Vec<SlugEvent>, ShortenerError>;

    /// Applies events to the aggregate rot instance
    fn apply(&mut self, event: &SlugEvent);
}

enum SlugCommand {
    CreateSlug { slug: Option<Slug>, url: Url },
    VisitSlug { slug: Slug },
}

trait EventListener {
    fn dispatch(&mut self, slug: &Slug, events: &[SlugEvent]);
}

trait Query<P, R> {
    fn execute(&self, params: P) -> R;
}

struct SlugAggregate { slug: Slug, url: Url, redirects: u64 }

struct EventStorage(Vec<EventEnvelope>);

#[derive(Clone)]
struct EventEnvelope { aggregate_id: String, event: SlugEvent }

impl SlugCommand {
    fn create(slug: Option<Slug>, url: Url) -> Self { SlugCommand::CreateSlug { slug: slug, url: url } }
    fn visit(slug: Slug) -> Self { SlugCommand::VisitSlug { slug: slug } }
}

impl SlugEvent {
    fn created(slug: Slug, url: Url) -> Self { SlugEvent::SlugCreated { slug: slug, url: url } }
    fn visited(slug: Slug, redirects: u64) -> Self { SlugEvent::SlugVisited { slug: slug, redirects: redirects } }
}

impl Cqrs {
    pub fn new(generator: Option<Box<dyn SlugGenerator + Sync + Send>>) -> Self {
        let generator: Box<dyn SlugGenerator + Sync + Send> = generator.unwrap_or_else(|| Box::new(DefaultSlugGenerator::new()));
        let store = EventStorage::new();
        let slug_list_query = SlugListQuery::new();
        let slug_details_query = SlugDetailsQuery::new();
        let used_slugs_query = UsedSlugsQuery::new();
        Self {
            generator: generator,
            store: store,
            slug_list_query: slug_list_query,
            slug_details_query: slug_details_query,
            used_slugs_query: used_slugs_query,
        }
    }

    pub fn process_new(&mut self, command: SlugCommand) -> Result<ShortLink, ShortenerError> {
        // not quite DDD-idiomatic
        let mut aggregate = SlugAggregate::new();
        self.process(Some(&mut aggregate), command)
    }

    pub fn load_and_process(&mut self, aggregate_id: &str, command: SlugCommand) -> Result<ShortLink, ShortenerError> {
        let mut aggregate = self.store.load(aggregate_id);
        self.process(aggregate.as_mut(), command)
    }

    pub fn process(&mut self, maybe_aggregate: Option<&mut SlugAggregate>, command: SlugCommand) -> Result<ShortLink, ShortenerError> {
        let aggregate = maybe_aggregate.ok_or(ShortenerError::SlugNotFound)?;
        let mut context = CommandContext { used_slugs_query: &self.used_slugs_query, generator: &mut self.generator };
        let events = aggregate.handle(&mut context, command)?;
        if events.is_empty() {
            return Ok(aggregate.into());
        }

        let events_slice = events.as_slice();
        self.store.commit(&aggregate.id(), events_slice);
        for event in events_slice {
            aggregate.apply(&event);
        }

        self.dispatch(&aggregate.slug, events_slice);
        Ok(aggregate.into())
    }

    fn dispatch(&mut self, slug: &Slug, events: &[SlugEvent]) {
        self.slug_list_query.dispatch(slug, events);
        self.slug_details_query.dispatch(slug, events);
        self.used_slugs_query.dispatch(slug, events);
    }

}

impl EventStorage {
    pub fn new() -> EventStorage {
        Self(vec![])
    }

    pub fn load(&self, aggregate_id: &str) -> Option<SlugAggregate> {
        let mut events = self.0.iter().filter(|e| aggregate_id == e.aggregate_id).peekable();
        if events.peek().is_none() {
            return None;
        }

        let mut aggregate = SlugAggregate::new();
        for envelope in events {
            aggregate.apply(&envelope.event);
        }

        return Some(aggregate);
    }

    pub fn commit(&mut self, aggregate_id: &str, events: &[SlugEvent]) {
        let entries = events.iter().map(|e| EventEnvelope { aggregate_id: aggregate_id.to_string(), event: e.clone() });
        self.0.extend(entries);
    }
}

use rand::distr::{Alphanumeric, SampleString};

pub trait SlugGenerator {
    fn generate(&mut self) -> String;
}

struct DefaultSlugGenerator;

impl DefaultSlugGenerator {
    pub fn new() -> Self { Self {} }
}

impl SlugGenerator for DefaultSlugGenerator{
    fn generate(&mut self) -> String {
        Alphanumeric.sample_string(&mut rand::rng(), 8)
    }
}

impl SlugAggregate {
    pub fn new() -> Self { Self { slug: Slug(String::new()), url: Url(String::new()), redirects: 0 } }
}

impl<'a> Into<ShortLink> for &'a mut SlugAggregate {
    fn into(self) -> ShortLink {
        ShortLink { slug: self.slug.clone(), url: self.url.clone() }
    }
}

fn validate_url(url: String) -> Result<String, ShortenerError> {
    match url::Url::parse(&url) {
        Ok(_) => Ok(url),
        Err(_) => Err(ShortenerError::InvalidUrl)
    }
}

use regex::Regex;
use lazy_static::lazy_static;
lazy_static! {
    static ref SLUG_REGEX: Regex  = Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9\\-]+$").unwrap();
}

fn validate_slug(slug: String) -> Result<String, ShortenerError> {
    if SLUG_REGEX.is_match(&slug) { Ok(slug) } else { Err(ShortenerError::InvalidUrl) }
}

impl Aggregate for SlugAggregate {
    fn id(&self) -> String { self.slug.0.clone() }

    fn handle<'a>(&mut self, context: &'a mut CommandContext, command: SlugCommand) -> Result<Vec<SlugEvent>, ShortenerError> {
        match command {
            SlugCommand::CreateSlug { slug, url } => {
                let valid_url = Url(validate_url(url.0)?);
                let actual_slug: Slug = match slug {
                    Option::None => {
                        let slug = loop {
                            let slug = context.generator.generate();
                            if !context.used_slugs_query.execute(&slug) {
                                break slug;
                            }
                        };
                        Ok(Slug(slug))
                    },
                    Some(slug) => {
                        let slug = validate_slug(slug.0)?;
                        if context.used_slugs_query.execute(&slug) {
                            Err(ShortenerError::SlugAlreadyInUse)
                        } else {
                            Ok(Slug(slug))
                        }
                    }
                }?;

                // HACK: ugly hack; refactoring required
                self.slug = actual_slug.clone();
                Ok(vec![SlugEvent::created(actual_slug, valid_url)])
            },
            SlugCommand::VisitSlug { slug } => {
                Ok(vec![SlugEvent::visited(slug, self.redirects + 1)])
            }
        }
    }

    fn apply(&mut self, event: &SlugEvent) {
        match event {
            SlugEvent::SlugCreated { slug, url } => {
                self.slug = slug.clone();
                self.url = url.clone();
                self.redirects = 0;
            },
            SlugEvent::SlugVisited { slug, redirects } => {
                self.redirects = *redirects;
            }
        }
    }
}

/// slug list query
use std::collections::HashSet;
struct SlugListQuery(HashSet<String>);
struct SlugListQueryParameters { skip: usize, take: usize }

impl SlugListQuery {
    pub fn new() -> Self { SlugListQuery(HashSet::new()) }
}

impl EventListener for SlugListQuery {
    fn dispatch(&mut self, slug: &Slug, events: &[SlugEvent]) {
        for event in events {
            match event {
                SlugEvent::SlugCreated { slug, url } => {
                    self.0.insert(slug.0.clone());
                },
                _ => {}
            }
        }
    }
}

impl Query<SlugListQueryParameters, Vec<String>> for SlugListQuery {
    fn execute(&self, params: SlugListQueryParameters) -> Vec<String> {
        let mut items: Vec<String> = self.0.iter().cloned().collect();
        items.sort();
        let mut items_iter: Box<dyn Iterator<Item = String>> = Box::new(items.into_iter());
        if 0 < params.skip {
            items_iter = Box::new(items_iter.skip(params.skip));
        }
        if 0 < params.take {
            items_iter = Box::new(items_iter.take(params.take));
        }

        items_iter.collect::<Vec<String>>()
    }
}

/// slug details query
use std::collections::HashMap;
struct SlugDetailsQuery(HashMap<String, Stats>);

impl SlugDetailsQuery {
    pub fn new() -> Self { SlugDetailsQuery(HashMap::new()) }
}

impl EventListener for SlugDetailsQuery {
    fn dispatch(&mut self, slug: &Slug, events: &[SlugEvent]) {
        for event in events {
            match event {
                SlugEvent::SlugCreated { slug, url } => {
                    let stats = Stats { link: ShortLink { slug: slug.clone(), url: url.clone() }, redirects: 0 };
                    self.0.insert(slug.0.clone(), stats );
                },
                SlugEvent::SlugVisited { slug, redirects } => {
                    self.0.entry(slug.0.clone()).and_modify(|stats| { stats.redirects = *redirects });
                }
            }
        }
    }
}

impl Query<Slug, Result<Stats, ShortenerError>> for SlugDetailsQuery {
    fn execute(&self, slug: Slug) -> Result<Stats, ShortenerError> {
        self.0.get(&slug.0).map(|v| v.clone()).ok_or(ShortenerError::SlugNotFound)
    }
}

/// used slugs query

struct UsedSlugsQuery(HashSet<String>);

impl UsedSlugsQuery {
    pub fn new() -> Self { UsedSlugsQuery(HashSet::new()) }
}

impl EventListener for UsedSlugsQuery {
    fn dispatch(&mut self, slug: &Slug, events: &[SlugEvent]) {
        for event in events {
            match event {
                SlugEvent::SlugCreated { slug, url } => {
                    self.0.insert(slug.0.clone());
                },
                _ => { }
            }
        }
    }
}

impl Query<&String, bool> for UsedSlugsQuery {
    fn execute(&self, slug: &String) -> bool {
        self.0.contains(slug)
    }
}

#[cfg(test)]
pub(crate) mod domain_tests {
    use crate::{DefaultSlugGenerator, SlugGenerator, validate_slug};

    #[test]
    fn check_validate_slug_accepts_correct_slugs() {
        assert!(validate_slug("this-is-a-valid-slug".to_string()).is_ok());
        assert!(validate_slug("1IlO0o".to_string()).is_ok());
    }

    #[test]
    fn check_validate_slug_rejects_invalid_slugs() {
        assert!(validate_slug("can-not-contain-,".to_string()).is_err());
        assert!(validate_slug("can-not-contain-.".to_string()).is_err());
        assert!(validate_slug("can-not-contain-!".to_string()).is_err());
        assert!(validate_slug("can-not-contain-#".to_string()).is_err());
        assert!(validate_slug("can-not-contain-?".to_string()).is_err());
        assert!(validate_slug("can not contain space".to_string()).is_err());
    }

    #[test]
    fn default_generator_generate() {
        let mut generator = DefaultSlugGenerator::new();
        for i in 1..100 {
            assert!(validate_slug(generator.generate()).is_ok());
        }
    }
}

#[cfg(test)]
pub(crate) mod cqrs_tests {
    use super::{
        Slug, Url, ShortenerError,
        Cqrs, SlugAggregate, SlugCommand, SlugGenerator,
        SlugEvent, SlugEvent::SlugCreated, SlugEvent::SlugVisited
    };
    use lazy_static::lazy_static;

    static VALID_URL_SLICE:&'static str = "http://github.com";
    static INVALID_URL_SLICE:&'static str = "not-an-url";

    lazy_static! {
        static ref VALID_URL: Url  = Url(VALID_URL_SLICE.to_string());
    }

    struct PredefinedSlugGenerator { values: Vec<String>, current: usize }

    impl PredefinedSlugGenerator {
        pub fn new(values: Vec<&str>) -> Self {
            Self { values: values.iter().map(|v| v.to_string()).collect(), current: 0}
        }
    }

    impl SlugGenerator for PredefinedSlugGenerator {
        fn generate(&mut self) -> String {
            let result = self.values[self.current].clone();
            self.current += 1;
            if self.values.len() >= self.current {
                self.current = 0;
            }

            result
        }
    }

    fn slug(slug: &str) -> Slug { Slug(slug.to_string()) }

    fn url(url: &str) -> Url { Url(url.to_string()) }

    fn create_cqrs() -> Cqrs {
        let generator = PredefinedSlugGenerator::new(vec!["aaa"]);
        let generator: Box<dyn SlugGenerator + Send + Sync> = Box::new(generator);
        Cqrs::new(Some(generator))
    }

    fn create_create_command(maybe_slug: Option<&str>, url_slice: &str) -> SlugCommand {
        SlugCommand::create(maybe_slug.map(slug), url(url_slice))
    }

    fn create_visit_command(slug_slice: &str) -> SlugCommand {
        SlugCommand::visit(slug(slug_slice))
    }

    fn create_created_event(slug_slice: &str, url_slice: &str) -> Vec<SlugEvent> {
        vec![SlugEvent::created(slug(slug_slice), url(url_slice))]
    }

    /// simulates slug creation
    fn create_slug(cqrs: &mut Cqrs, slug_slice: &str, url_slice: &str) {
        let slug = slug(slug_slice);
        let events = create_created_event(slug_slice, url_slice);
        cqrs.store.commit(slug_slice, &events);
        cqrs.dispatch(&slug, &events);
    }

    #[test]
    fn generate_slug() {
        let mut cqrs = create_cqrs();
        let command = create_create_command(None, VALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command).expect("should create slug");
        assert_eq!(aggregate.url.0, VALID_URL_SLICE);
        assert_eq!(cqrs.store.0.len(), 1);
        assert_eq!(cqrs.store.0[0].aggregate_id, "aaa");
        let expected_slug = slug("aaa");
        let expected_url = VALID_URL.clone();
        assert!(matches!(&cqrs.store.0[0].event, SlugCreated { slug: expected_slug, url: expected_url }));
        let slug_aggregate = cqrs.store.load("aaa").expect("slug should be created");
        assert!(matches!(slug_aggregate, SlugAggregate { slug: expected_slug, url: expected_url, redirects: 0 }));
    }

    #[test]
    fn generate_slug_invalid_url() {
        let mut cqrs = create_cqrs();
        let command = create_create_command(None, INVALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command);
        assert_eq!(result, Err(ShortenerError::InvalidUrl));
        assert_eq!(cqrs.store.0.len(), 0);
        let slug_aggregate = cqrs.store.load("aaa");
        assert!(matches!(slug_aggregate, None));
    }

    #[test]
    fn create_with_slug() {
        let mut cqrs = create_cqrs();
        let command = create_create_command(Some("valid-slug"), VALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command).expect("slug should be created");
        assert_eq!(result.slug, slug("valid-slug"));
        assert_eq!(cqrs.store.0.len(), 1);
        let slug_aggregate = cqrs.store.load("valid-slug").expect("slug should be created");
        let expected_slug = slug("valid-slug");
        let expected_url = VALID_URL.clone();
        assert!(matches!(slug_aggregate, SlugAggregate { slug: expected_slug, url: expected_url, redirects: 0 }));
    }

    #[test]
    fn create_with_existing_slug() {
        let valid_slug = "valid-slug";
        let mut cqrs = create_cqrs();
        // simulate slug cration
        let slug = slug(valid_slug);
        cqrs.dispatch(&slug, &vec![SlugEvent::created(slug.clone(), VALID_URL.clone())]);
        // create the same slug
        let command = create_create_command(Some(valid_slug), VALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command);
        assert_eq!(result, Err(ShortenerError::SlugAlreadyInUse));
    }

    #[test]
    fn create_with_invalid_slug() {
        let mut cqrs = create_cqrs();
        let command = create_create_command(Some("!! not a slug"), VALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command);
        assert_eq!(result, Err(ShortenerError::InvalidUrl));
    }

    #[test]
    fn create_with_slug_and_invalid_url() {
        let mut cqrs = create_cqrs();
        let command = create_create_command(Some("valid-slug"), INVALID_URL_SLICE);
        let mut aggregate = SlugAggregate::new();
        let result = cqrs.process(Some(&mut aggregate), command);
        assert_eq!(result, Err(ShortenerError::InvalidUrl));
    }

    #[test]
    fn visit_slug() {
        let valid_slug = "valid-slug";
        let mut cqrs = create_cqrs();
        create_slug(&mut cqrs, valid_slug, VALID_URL_SLICE);
        // invoke 'visit' command
        let command = create_visit_command(valid_slug);
        let result = cqrs.load_and_process(&valid_slug, command).expect("should visit slug");
        assert_eq!(result.slug.0, valid_slug);
        assert_eq!(result.url.0, VALID_URL_SLICE);
        assert_eq!(cqrs.store.0.len(), 2);
        assert_eq!(cqrs.store.0[0].aggregate_id, valid_slug);
        assert_eq!(cqrs.store.0[1].aggregate_id, valid_slug);
        let expected_url = VALID_URL.clone();
        assert!(matches!(&cqrs.store.0[0].event, SlugCreated { slug: expected_slug, url: expected_url }));
        assert!(matches!(&cqrs.store.0[1].event, SlugVisited { slug: expected_slug, redirects: 1 }));
        let aggregate = cqrs.store.load(valid_slug).expect("slug should exist");
        assert!(matches!(aggregate, SlugAggregate { slug: _slug, url: expected_url, redirects: 1 }));
    }

    #[test]
    fn visit_invalid_slug() {
        let valid_slug = "valid-slug";
        let mut cqrs = create_cqrs();
        create_slug(&mut cqrs, valid_slug, VALID_URL_SLICE);
        // invoke 'visit' command
        let command = create_visit_command("!invalid slug");
        let result = cqrs.load_and_process("missing-slug", command);
        assert!(result.is_err());
        assert!(matches!(result, Err(ShortenerError::SlugNotFound)));
    }

    #[test]
    fn visit_missing_slug() {
        let valid_slug = "valid-slug";
        let mut cqrs = create_cqrs();
        create_slug(&mut cqrs, valid_slug, VALID_URL_SLICE);
        // invoke 'visit' command
        let command = create_visit_command("missing");
        let result = cqrs.load_and_process("missing-slug", command);
        assert!(result.is_err());
        assert!(matches!(result, Err(ShortenerError::SlugNotFound)));
    }
}

/// api implementation
use actix_web::{ web, get, post, http::header, App, HttpServer, HttpResponse, Responder };
use std::{ env, sync::Mutex };

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
  skip: Option<usize>,
  take: Option<usize>,
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