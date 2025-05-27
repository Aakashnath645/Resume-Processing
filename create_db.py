import sqlite3

def create_database():
    try:
        conn = sqlite3.connect('resume_analysis.db')
        c = conn.cursor()
        
        # Drop existing tables if recreating
        c.execute('DROP TABLE IF EXISTS analyses')
        
        # Create analyses table with all required columns
        c.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                match_percentage REAL NOT NULL,
                suitable TEXT NOT NULL,
                detailed_analysis TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resume_file_name TEXT,
                ai_scores TEXT,
                ats_score REAL
            )
        ''')
        
        # Create indexes for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_candidate ON analyses(candidate_name)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_department ON analyses(department)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses(timestamp)')
        
        conn.commit()
        print("Database created successfully with all required columns!")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()
