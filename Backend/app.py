from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
from models.model_manager import ModelManager

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE = 'database/learning.db'

# Initialize Model Manager
model_manager = ModelManager()

# Load default model at startup (TinyLlama is faster for development)
print("Initializing AI model...")
model_manager.load_model('tinyllama')  # Change to 'phi3-mini' for better quality

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            grade_level INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            plan_content TEXT,
            subjects TEXT,
            model_used TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            lesson_id TEXT,
            completed BOOLEAN DEFAULT 0,
            score INTEGER,
            completed_at DATETIME,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Database helper
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.get_current_model()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available AI models"""
    return jsonify({
        'available': model_manager.get_available_models(),
        'current': model_manager.get_current_model()
    })

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.json
    model_key = data.get('model')
    
    if not model_key:
        return jsonify({'error': 'Model key required'}), 400
    
    try:
        success = model_manager.load_model(model_key)
        if success:
            return jsonify({
                'success': True,
                'current_model': model_manager.get_current_model()
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['POST'])
def create_student():
    data = request.json
    name = data.get('name')
    grade_level = data.get('gradeLevel')
    
    if not name or not grade_level:
        return jsonify({'error': 'Name and grade level are required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO students (name, grade_level) VALUES (?, ?)',
        (name, grade_level)
    )
    
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'studentId': student_id,
        'name': name,
        'gradeLevel': grade_level
    }), 201

@app.route('/api/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
    student = cursor.fetchone()
    conn.close()
    
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    
    return jsonify(dict(student))

@app.route('/api/generate-study-plan', methods=['POST'])
def generate_study_plan():
    data = request.json
    student_id = data.get('studentId')
    grade_level = data.get('gradeLevel')
    subjects = data.get('subjects', [])
    current_progress = data.get('currentProgress', 'Beginning')
    
    if not student_id or not grade_level:
        return jsonify({'error': 'Student ID and grade level are required'}), 400
    
    # Create educational prompt
    system_context = "You are an expert K-5 educational assistant. Create engaging, age-appropriate study plans."
    
    user_prompt = f"""Create a personalized study plan for a grade {grade_level} student.

Focus areas: {', '.join(subjects)}
Current progress: {current_progress}

Generate a structured weekly plan with:
1. Daily learning objectives (Monday-Friday)
2. Specific Khan Academy topics to cover
3. Practice exercises for each day
4. End-of-week assessment

Keep it simple, fun, and age-appropriate for elementary students. Use encouraging language."""

    # Combine for the model
    full_prompt = f"{system_context}\n\n{user_prompt}"
    
    try:
        print(f"Generating study plan with {model_manager.get_current_model()}...")
        
        # Generate with embedded model
        plan_content = model_manager.generate(
            full_prompt,
            max_length=800,
            temperature=0.7
        )
        
        if not plan_content:
            return jsonify({'error': 'Failed to generate study plan'}), 500
        
        # Save to database
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO study_plans (student_id, plan_content, subjects, model_used) VALUES (?, ?, ?, ?)',
            (student_id, plan_content, ','.join(subjects), model_manager.get_current_model())
        )
        
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'planId': plan_id,
            'content': plan_content,
            'modelUsed': model_manager.get_current_model()
        })
        
    except Exception as e:
        print(f"Error generating study plan: {e}")
        return jsonify({
            'error': 'Failed to generate study plan',
            'details': str(e)
        }), 500

@app.route('/api/students/<int:student_id>/study-plans', methods=['GET'])
def get_study_plans(student_id):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT * FROM study_plans WHERE student_id = ? ORDER BY created_at DESC',
        (student_id,)
    )
    
    plans = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(plans)

@app.route('/api/progress', methods=['POST'])
def save_progress():
    data = request.json
    student_id = data.get('studentId')
    lesson_id = data.get('lessonId')
    completed = data.get('completed', False)
    score = data.get('score')
    
    if not student_id or not lesson_id:
        return jsonify({'error': 'Student ID and lesson ID are required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        '''INSERT INTO progress (student_id, lesson_id, completed, score, completed_at)
           VALUES (?, ?, ?, ?, datetime('now'))''',
        (student_id, lesson_id, 1 if completed else 0, score)
    )
    
    progress_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({'progressId': progress_id}), 201

@app.route('/api/students/<int:student_id>/progress', methods=['GET'])
def get_progress(student_id):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT * FROM progress WHERE student_id = ? ORDER BY completed_at DESC',
        (student_id,)
    )
    
    progress = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(progress)

@app.route('/api/students/<int:student_id>/stats', methods=['GET'])
def get_student_stats(student_id):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT COUNT(*) as total FROM progress WHERE student_id = ?',
        (student_id,)
    )
    total_lessons = cursor.fetchone()['total']
    
    cursor.execute(
        'SELECT COUNT(*) as completed FROM progress WHERE student_id = ? AND completed = 1',
        (student_id,)
    )
    completed_lessons = cursor.fetchone()['completed']
    
    cursor.execute(
        'SELECT AVG(score) as avg_score FROM progress WHERE student_id = ? AND score IS NOT NULL',
        (student_id,)
    )
    avg_score = cursor.fetchone()['avg_score'] or 0
    
    conn.close()
    
    return jsonify({
        'totalLessons': total_lessons,
        'completedLessons': completed_lessons,
        'averageScore': round(avg_score, 2),
        'completionRate': round((completed_lessons / total_lessons * 100) if total_lessons > 0 else 0, 2)
    })

if __name__ == '__main__':
    os.makedirs('database', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    init_db()
    
    port = int(os.getenv('PORT', 5000))
    
    print(f'\n🚀 Flask backend running on http://localhost:{port}')
    print(f'🤖 AI Model: {model_manager.get_current_model()}')
    print(f'📚 Ready to generate offline study plans!\n')
    
    app.run(debug=True, host='0.0.0.0', port=port)