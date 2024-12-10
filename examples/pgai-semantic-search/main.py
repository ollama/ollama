import psycopg2
import time
from psycopg2.extras import Json
import requests

def wait_for_postgres(conn_params, max_attempts=30, delay_seconds=1):
    """Wait for PostgreSQL to be ready to accept connections."""
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**conn_params)
            conn.close()
            return
        except psycopg2.OperationalError:
            if attempt + 1 == max_attempts:
                raise
            time.sleep(delay_seconds)


def wait_for_ollama(max_attempts=30, delay_seconds=1):
    """Wait for Ollama to be ready and download the required model."""
    ollama_url = "http://localhost:11434"

    print("Waiting for Ollama to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if attempt + 1 == max_attempts:
                raise
            time.sleep(delay_seconds)

    print("Downloading nomic-embed-text model...")
    response = requests.post(
        f"{ollama_url}/api/pull",
        json={"name": "nomic-embed-text"}
    )

    if response.status_code != 200:
        raise Exception("Failed to download model")

def main():
    # Connection parameters
    conn_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432"
    }

    # Wait for services to be ready
    wait_for_ollama()
    print("Waiting for PostgreSQL to be ready...")
    wait_for_postgres(conn_params)

    # Connect to the database
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    cur = conn.cursor()

    print("Setting up database...")

    # Enable pgai extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS ai CASCADE;")
    
    cur.execute("SELECT * from ai.vectorizer;")
    
    results = cur.fetchall()
    
    if len(results) > 0:
        print("Vectorizer already exists, dropping it...")
        cur.execute("SELECT * from ai.drop_vectorizer(1, drop_all => true);")

    # Create blog table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS blog (
            id SERIAL PRIMARY KEY,
            title TEXT,
            authors TEXT,
            contents TEXT,
            metadata JSONB
        );
    """)

    # Insert sample data
    sample_data = [
        ('Getting Started with PostgreSQL', 'John Doe',
         'PostgreSQL is a powerful, open source object-relational database system...',
         {'tags': ['database', 'postgresql', 'beginner'], 'read_time': 5, 'published_date': '2024-03-15'}),

        ('10 Tips for Effective Blogging', 'Jane Smith, Mike Johnson',
         'Blogging can be a great way to share your thoughts and expertise...',
         {'tags': ['blogging', 'writing', 'tips'], 'read_time': 8, 'published_date': '2024-03-20'}),

        ('The Future of Artificial Intelligence', 'Dr. Alan Turing',
         'As we look towards the future, artificial intelligence continues to evolve...',
         {'tags': ['AI', 'technology', 'future'], 'read_time': 12, 'published_date': '2024-04-01'}),

        ('Healthy Eating Habits for Busy Professionals', 'Samantha Lee',
         'Maintaining a healthy diet can be challenging for busy professionals...',
         {'tags': ['health', 'nutrition', 'lifestyle'], 'read_time': 6, 'published_date': '2024-04-05'}),

        ('Introduction to Cloud Computing', 'Chris Anderson',
         'Cloud computing has revolutionized the way businesses operate...',
         {'tags': ['cloud', 'technology', 'business'], 'read_time': 10, 'published_date': '2024-04-10'})
    ]

    cur.execute("TRUNCATE blog CASCADE;")  # Clear existing data
    for title, authors, contents, metadata in sample_data:
        cur.execute(
            "INSERT INTO blog (title, authors, contents, metadata) VALUES (%s, %s, %s, %s)",
            (title, authors, contents, Json(metadata))
        )

    print("Creating vectorizer...")
    # Create vectorizer
    cur.execute("""
        SELECT ai.create_vectorizer(
            'blog'::regclass,
            destination => 'blog_contents_embeddings',
            embedding => ai.embedding_ollama('nomic-embed-text', 768),
            chunking => ai.chunking_recursive_character_text_splitter('contents')
        );
    """)

    print("Waiting for vectorizer to process data...")
    time.sleep(10)  # Give the vectorizer some time to process

    print("\nPerforming semantic search for 'good food'...")
    # Perform semantic search
    cur.execute("""
        SELECT
            b.title,
            b.contents,
            be.chunk,
            be.embedding <=> ai.ollama_embed('nomic-embed-text', 'good food') as distance
        FROM blog_contents_embeddings be
        JOIN blog b ON b.id = be.id
        ORDER BY distance
        LIMIT 3;
    """)

    results = cur.fetchall()
    print("\nTop 3 most relevant results:")
    print("-" * 80)
    for title, content, chunk, distance in results:
        print(f"Title: {title}")
        print(f"Distance: {distance}")
        print(f"Content snippet: {chunk[:100]}...")
        print("-" * 80)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
