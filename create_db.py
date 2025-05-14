import sqlite3

def create_database():
    try:
        conn = sqlite3.connect('resume_analysis.db')
        c = conn.cursor()
        
        # Drop existing tables
        c.execute('DROP TABLE IF EXISTS analyses')
        
        # Create analyses table with updated columns
        c.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                match_percentage REAL NOT NULL,
                suitable TEXT NOT NULL,
                detailed_analysis TEXT,
                ai_scores TEXT,
                ats_score REAL,
                resume_file_name TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database created successfully!")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_database()