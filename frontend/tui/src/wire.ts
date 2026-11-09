// The transport and the wire types are the web client's api.ts. Node resolves
// the shared file through the "#core/api" subpath import declared in
// frontend/package.json, so the TUI never reaches into web/ by path.
export * from '#core/api';
