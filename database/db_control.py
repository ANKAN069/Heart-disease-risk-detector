import sqlite3
import os

def run_sqlite_simulator():
    print("=== Interactive SQLite Simulator ===")
    db_name = input("Enter .db filename (e.g., test.db): ").strip()
    
    if not db_name:
        db_name = "simulator_default.db"
        print(font_style:=f"No name entered. Using default: {db_name}")

    try:
        # Connect to the database file (creates it if it doesn't exist)
        connection = sqlite3.connect(db_name)
        cursor = connection.cursor()
        print(f"Connected successfully to '{db_name}'.")
        print("Type your SQL queries below. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            # Get multi-line or single-line query from user
            query = input("sqlite> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("Closing database connection. Goodbye!")
                break
            
            # The try-except block prevents the simulator from crashing on bad SQL syntax
            try:
                cursor.execute(query)
                
                # If it's a SELECT query, fetch and display results
                if cursor.description:
                    colnames = [desc[0] for desc in cursor.description]
                    print(" | ".join(colnames))
                    print("-" * (len(" | ".join(colnames)) + 4))
                    
                    rows = cursor.fetchall()
                    if not rows:
                        print("(No rows returned)")
                    for row in rows:
                        print(" | ".join(str(item) for item in row))
                else:
                    # For INSERT, UPDATE, DELETE, CREATE, DROP
                    connection.commit()
                    print(f"Query executed successfully. Rows affected: {cursor.rowcount}")
                    
            except sqlite3.Error as e:
                # Catch SQL errors (syntax errors, missing tables, etc.) without crashing
                print(f"SQL Error: {e}")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Unexpected Error: {e}")
                
            print() # Print an empty line for readability

    except sqlite3.Error as e:
        print(f"Could not connect to database: {e}")
    finally:
        # Clean up and close connection safely
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    run_sqlite_simulator()
