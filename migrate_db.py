import sqlite3

def migrate_database():
    try:
        conn = sqlite3.connect('resume_analysis.db')
        c = conn.cursor()
        
        # Get existing columns
        c.execute("PRAGMA table_info(analyses)")
        columns = {col[1] for col in c.fetchall()}
        
        # Add missing columns if they don't exist
        if 'ai_scores' not in columns:
            c.execute('ALTER TABLE analyses ADD COLUMN ai_scores TEXT')
            
        if 'ats_score' not in columns:
            c.execute('ALTER TABLE analyses ADD COLUMN ats_score REAL')
            
        if 'timestamp' not in columns:
            c.execute('ALTER TABLE analyses ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        
        conn.commit()
        print("Database migration completed successfully!")
        
    except sqlite3.Error as e:
        print(f"Migration error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
