import weaviate
import weaviate.classes as wvc


def main():
    CLASS_NAME = "Article"
    
    # Connect to local Weaviate instance
    client = weaviate.connect_to_local()
    print("Connected to local Weaviate instance")

    try:
        # Clean up any existing data
        client.collections.delete_all()

        # Create schema
        client.collections.create(
            name=CLASS_NAME,
            description="An Article class to store a text",
            properties=[
                wvc.config.Property(
                    name="author",
                    data_type=wvc.config.DataType.TEXT,
                    description="The name of the author",
                ),
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                    description="The text content",
                ),
            ],
        )
        print("Created schema")

        # Get collection
        collection = client.collections.get(CLASS_NAME)
        print(f"Retrieved collection: {collection.name}")

        # Create single object
        uuid = collection.data.insert(
            {
                "author": "Robert",
                "text": "Once upon a time, someone wrote a book...",
            }
        )
        print(f"Created object of UUID: {uuid}")

        # Fetch object by ID
        obj = collection.query.fetch_object_by_id(uuid)
        print(f"Retrieved obj: {obj}")

        # Create batch
        objs = [
            {
                "author": "Robert",
                "text": "Once upon a time, R. wrote a book...",
            },
            {
                "author": "Johnson",
                "text": "Once upon a time, J. wrote some news...",
            },
            {
                "author": "Maverick",
                "text": "Never again, M. will write a book...",
            },
            {
                "author": "Wilson",
                "text": "Lost in the island, W. did not write anything...",
            },
            {
                "author": "Ludwig",
                "text": "As king, he ruled...",
            },
        ]
        with collection.batch.dynamic() as batch:
            for obj in objs:
                batch.add_object(properties=obj)
        print("Created batch objects")

        # Query
        result = collection.query.fetch_objects(
            limit=5,
            return_properties=["author", "text"],
        )
        print(f"Query result: {result}")

        # Aggregate (not instrumented yet)
        aggregate_result = collection.aggregate.over_all(total_count=True)
        print(f"Aggregate result: {aggregate_result}")

        # Delete collection
        client.collections.delete(CLASS_NAME)
        print("Deleted schema")

    finally:
        client.close()


if __name__ == "__main__":
    main()
