SPAN_NAME_PREFIX: str = "db.weaviate"

CONNECTION_WRAPPING: list[dict[str, str]] = [
    {"module": "weaviate", "name": "connect_to_local"},
    {"module": "weaviate", "name": "connect_to_weaviate_cloud"},
    {"module": "weaviate", "name": "connect_to_custom"},
]

MAPPING_V3: list[dict[str, str]] = [
    # Schema operations
    {
        "module": "weaviate.schema",
        "name": "Schema",
        "function": "get",
        "span_name": "schema.get",
    },
    {
        "module": "weaviate.schema",
        "name": "Schema",
        "function": "create_class",
        "span_name": "schema.create_class",
    },
    {
        "module": "weaviate.schema",
        "name": "Schema",
        "function": "create",
        "span_name": "schema.create",
    },
    {
        "module": "weaviate.schema",
        "name": "Schema",
        "function": "delete_class",
        "span_name": "schema.delete_class",
    },
    {
        "module": "weaviate.schema",
        "name": "Schema",
        "function": "delete_all",
        "span_name": "schema.delete_all",
    },
    # Data CRUD operations
    {
        "module": "weaviate.data.crud_data",
        "name": "DataObject",
        "function": "create",
        "span_name": "data.crud_data.create",
    },
    {
        "module": "weaviate.data.crud_data",
        "name": "DataObject",
        "function": "validate",
        "span_name": "data.crud_data.validate",
    },
    {
        "module": "weaviate.data.crud_data",
        "name": "DataObject",
        "function": "get",
        "span_name": "data.crud_data.get",
    },
    # Batch operations
    {
        "module": "weaviate.batch.crud_batch",
        "name": "Batch",
        "function": "add_data_object",
        "span_name": "batch.crud_batch.add_data_object",
    },
    {
        "module": "weaviate.batch.crud_batch",
        "name": "Batch",
        "function": "flush",
        "span_name": "batch.crud_batch.flush",
    },
    # GraphQL query operations
    {
        "module": "weaviate.gql.query",
        "name": "Query",
        "function": "get",
        "span_name": "gql.query.get",
    },
    {
        "module": "weaviate.gql.query",
        "name": "Query",
        "function": "aggregate",
        "span_name": "gql.query.aggregate",
    },
    {
        "module": "weaviate.gql.query",
        "name": "Query",
        "function": "raw",
        "span_name": "gql.query.raw",
    },
    {
        "module": "weaviate.gql.get",
        "name": "GetBuilder",
        "function": "do",
        "span_name": "gql.query.get.do",
    },
]


MAPPING_V4: list[dict[str, str]] = [
    {
        "module": "weaviate.collections.queries.near_text.query",
        "name": "_NearTextQuery",
        "function": "near_text",
        "span_name": "collections.query.near_text",
    },
    {
        "module": "weaviate.collections.queries.near_vector.query",
        "name": "_NearVectorQuery",
        "function": "near_vector",
        "span_name": "collections.query.near_vector",
    },
    {
        "module": "weaviate.collections.queries.fetch_objects.query",
        "name": "_FetchObjectsQuery",
        "function": "fetch_objects",
        "span_name": "collections.query.fetch_objects",
    },
    {
        "module": "weaviate.collections.grpc.query",
        "name": "_QueryGRPC",
        "function": "get",
        "span_name": "collections.query.get",
    },
    {
        "module": "weaviate.collections.data",
        "name": "_DataCollection",
        "function": "insert",
        "span_name": "collections.data.insert",
    },
    {
        "module": "weaviate.collections.data",
        "name": "_DataCollection",
        "function": "replace",
        "span_name": "collections.data.replace",
    },
    {
        "module": "weaviate.collections.data",
        "name": "_DataCollection",
        "function": "update",
        "span_name": "collections.data.update",
    },
    # Collections
    {
        "module": "weaviate.collections.collections",
        "name": "_Collections",
        "function": "get",
        "span_name": "collections.get",
    },
    {
        "module": "weaviate.collections.collections",
        "name": "_Collections",
        "function": "create",
        "span_name": "collections.create",
    },
    {
        "module": "weaviate.collections.collections",
        "name": "_Collections",
        "function": "delete",
        "span_name": "collections.delete",
    },
    {
        "module": "weaviate.collections.collections",
        "name": "_Collections",
        "function": "delete_all",
        "span_name": "collections.delete_all",
    },
    {
        "module": "weaviate.collections.collections",
        "name": "_Collections",
        "function": "create_from_dict",
        "span_name": "collections.create_from_dict",
    },
    # Batch
    {
        "module": "weaviate.collections.batch.collection",
        "name": "_BatchCollection",
        "function": "add_object",
        "span_name": "collections.batch.add_object",
    },
]