import sqlite3
import time

class SQLHandler:
    def __init__(self, media_files_db):
        self.media_files_db = media_files_db
        # Create a connection to the SQLite database
        self.conn = sqlite3.connect(media_files_db, check_same_thread=False)

        with self.conn as conn: 
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS media_files
                        (file_id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        timestamp INT NOT NULL,
                        status TEXT)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS logo_placement_tasks
                        (video_id TEXT NOT NULL,
                        logo_id TEXT NOT NULL,
                        timestamp INT NOT NULL)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS logo_placement_results
                        (video_id TEXT NOT NULL,
                        logo_id TEXT NOT NULL,
                        task_timestamp INT NOT NULL,
                        result_video_path TEXT NOT NULL)''')
            conn.commit()

    
    def action_query(self, query):
        print('set_operation')
        print(query)
        print()
        with self.conn as conn: 
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()

    def get_query_result(self, query):
        print('get_operation')
        print(query)
        print()
        with self.conn as conn: 
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def select_all(self, table='media_files'):
        query = f"Select * from {table}"
        return self.get_query_result(query)

    def insert_file(self, file_id, client_id, file_type, file_path, status='ok'):
        timestamp = int(time.time())
        args = (file_id, client_id, file_type, file_path, timestamp, status)
        query = f"INSERT INTO media_files (file_id, client_id, file_type, file_path, timestamp, status) VALUES {args}"
        self.action_query(query)

    
    def check_status(self, file_id):
        query = f"Select status from media_files where file_id == '{file_id}' LIMIT 1"
        return self.get_query_result(query)
    
    def get_videos_to_process(self, num_videos):
        query = f'''
        SELECT 
        file_id 
        FROM media_files 
        WHERE status = 'need_process'
        AND file_type = 'video'
        ORDER BY timestamp
        LIMIT {num_videos}'''

        videos = self.get_query_result(query)

        return videos
    
    def set_videos_status_ready(self, video_names):
        video_names_quota = map(lambda x: "'" + x + "'", video_names)
        video_names_string = ','.join(video_names_quota)
        query = f"UPDATE media_files SET status = 'ready' WHERE file_id IN ({video_names_string})"
        self.action_query(query)
        
    def insert_logo_place_task(self, video_id, logo_id):
        timestamp = int(time.time())
        query = f"INSERT INTO logo_placement_tasks (video_id, logo_id, timestamp) VALUES {(video_id, logo_id, timestamp)}"
        self.action_query(query)
    
    def get_oldest_logo_placement_tasks(self, num_tasks):
        query = f"SELECT video_id, logo_id, timestamp FROM logo_placement_tasks ORDER BY timestamp LIMIT {num_tasks}"
        logos = self.get_query_result(query)
        return logos

    def remove_placement_task(self, video_id, logo_id):
        query = f"DELETE FROM logo_placement_tasks WHERE video_id = '{video_id}' AND logo_id = '{logo_id}'"
        self.action_query(query)

    def insert_logo_placement_result(self, video_id, logo_id, timestamp, result_video_path):
        query = f'''
        INSERT INTO logo_placement_results
        (video_id, logo_id, task_timestamp, result_video_path) 
        VALUES {(video_id, logo_id, timestamp, result_video_path)}'''
        self.action_query(query)

    def get_newest_user_ready_video(self, client_id):
        query = f'''
            SELECT 
            file_id 
            FROM media_files 
            WHERE status = 'ready'
            AND client_id = '{client_id}'
            AND file_type = 'video'
            ORDER BY timestamp DESC
            LIMIT 1'''
        return self.get_query_result(query)

    def get_newest_user_ready_logo(self, client_id):
        query = f'''
            SELECT 
            file_id 
            FROM media_files 
            WHERE status = 'ready'
            AND client_id = '{client_id}'
            AND file_type = 'image'
            ORDER BY timestamp DESC
            LIMIT 1'''
        return self.get_query_result(query)

    def get_logo_placement_result(self, video_id, logo_id):
        query = f'''
            SELECT 
            result_video_path 
            FROM logo_placement_results 
            WHERE 
            video_id = '{video_id}'
            AND logo_id = '{logo_id}'
            LIMIT 1'''
        return self.get_query_result(query)


    def check_placement_task(self, video_id, logo_id):
        query = f'''
            SELECT 
            timestamp
            FROM logo_placement_tasks 
            WHERE 
            video_id = '{video_id}'
            AND logo_id = '{logo_id}'
            LIMIT 1'''
        return self.get_query_result(query)


    def get_placement_task_queue_num(self, video_id, logo_id):
        query = f'''
            SELECT 
            video_id, logo_id
            FROM logo_placement_tasks 
            ORDER BY timestamp'''
        
        result = self.get_query_result(query)
        for idx, (row_video_id, row_logo_id) in enumerate(result):
            if row_video_id == video_id and row_logo_id == logo_id:
                return idx+1
        return -1
