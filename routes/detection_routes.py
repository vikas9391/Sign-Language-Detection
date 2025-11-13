from flask import Blueprint, Response, render_template, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta
import logging
import atexit
import time
from sqlalchemy import func

from camera.detector import VideoCamera, get_latest_label
from models.detection_model import Detection
from database.db_setup import db

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__, url_prefix='/detection')

# Store active camera instances per user
active_cameras = {}


def get_camera(user_id):
    """Get or create camera instance for specific user"""
    global active_cameras
    
    # Clean up old camera if exists
    if user_id in active_cameras:
        try:
            logger.info(f"🧹 Cleaning up existing camera for user {user_id}")
            active_cameras[user_id].stop()
            time.sleep(0.3)  # Give time for cleanup
        except Exception as e:
            logger.warning(f"⚠️ Error stopping old camera for user {user_id}: {e}")
    
    # Create new camera instance
    try:
        logger.info(f"📹 Initializing camera for user {user_id}...")
        camera = VideoCamera()
        active_cameras[user_id] = camera
        return camera
    except Exception as e:
        logger.error(f"❌ Camera initialization error for user {user_id}: {e}")
        raise


@detection_bp.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route - returns multipart response"""
    try:
        user_id = current_user.id
        logger.info(f"📺 Starting video feed for user {user_id}")
        cam = get_camera(user_id)
        return Response(
            cam.get_frame(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"❌ Video feed error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@detection_bp.route('/get_label')
@login_required
def get_label():
    """Get the latest detected label as JSON and save to database"""
    try:
        label, confidence = get_latest_label()
        
        # Debug logging - only log when there's a label
        if label:
            logger.debug(f"📤 Sending to frontend: label={label}, confidence={confidence:.2f}")
        
        # Save detection to database if confidence is high enough
        if label and confidence > 0.75:
            try:
                # Check if this sign was recently saved (avoid duplicates within 2 seconds)
                recent_time = datetime.utcnow() - timedelta(seconds=2)
                
                recent_detection = Detection.query.filter(
                    Detection.user_id == current_user.id,
                    Detection.detected_sign == label,
                    Detection.timestamp >= recent_time
                ).first()
                
                if not recent_detection:
                    detection = Detection()
                    detection.user_id = current_user.id
                    detection.detected_sign = label
                    db.session.add(detection)
                    db.session.commit()
                    logger.info(f"💾 Saved detection: {label} for user {current_user.id}")
            except Exception as e:
                logger.error(f"❌ Error saving detection: {e}")
                db.session.rollback()
        
        # IMPORTANT: Always return proper JSON response
        response_data = {
            "label": label,
            "confidence": float(confidence) if confidence else 0.0
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting label: {e}", exc_info=True)
        return jsonify({
            "label": "",
            "confidence": 0.0
        }), 200  # Return 200 even on error to prevent frontend issues


@detection_bp.route('/')
@detection_bp.route('/detect')
@login_required
def detect():
    """Main detection page"""
    return render_template('home.html')


@detection_bp.route('/stop')
@detection_bp.route('/stop_camera')
@login_required
def stop_camera():
    """Stop and release camera for current user"""
    global active_cameras
    
    try:
        user_id = current_user.id
        
        if user_id in active_cameras:
            try:
                logger.info(f"🛑 Stopping camera for user {user_id}")
                active_cameras[user_id].stop()
                time.sleep(0.2)  # Allow cleanup time
                del active_cameras[user_id]
                logger.info(f"✅ Camera stopped for user {user_id}")
                return jsonify({
                    "success": True,
                    "status": "Camera stopped"
                })
            except Exception as e:
                logger.error(f"❌ Error stopping camera for user {user_id}: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        return jsonify({
            "success": True,
            "status": "Camera was not running"
        })
    except Exception as e:
        logger.error(f"❌ Error in stop_camera: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@detection_bp.route('/detection_history')
@login_required
def detection_history():
    """Get user's detection history"""
    try:
        # Get recent detections (last 100)
        detections = Detection.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Detection.timestamp.desc()
        ).limit(100).all()
        
        history = [
            {
                "id": d.id,
                "sign": d.detected_sign,
                "timestamp": d.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            for d in detections
        ]
        
        return jsonify({
            "success": True,
            "history": history
        })
    except Exception as e:
        logger.error(f"❌ Error fetching detection history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@detection_bp.route('/clear_history', methods=["POST"])
@login_required
def clear_history():
    """Clear user's detection history"""
    try:
        deleted_count = Detection.query.filter_by(
            user_id=current_user.id
        ).delete()
        
        db.session.commit()
        logger.info(f"🗑️ Cleared {deleted_count} detections for user {current_user.id}")
        
        return jsonify({
            "success": True,
            "message": f"Cleared {deleted_count} detection(s) from history"
        })
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@detection_bp.route('/detection_stats')
@login_required
def detection_stats():
    """Get user's detection statistics"""
    try:
        # Total detections
        total_detections = Detection.query.filter_by(
            user_id=current_user.id
        ).count()
        
        # Get unique signs detected
        unique_signs = db.session.query(
            Detection.detected_sign
        ).filter_by(
            user_id=current_user.id
        ).distinct().count()
        
        # Get most recent detection
        recent_detection = Detection.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Detection.timestamp.desc()
        ).first()
        
        # Get most common sign
        most_common = db.session.query(
            Detection.detected_sign,
            func.count(Detection.id).label('count')
        ).filter_by(
            user_id=current_user.id
        ).group_by(
            Detection.detected_sign
        ).order_by(
            func.count(Detection.id).desc()
        ).first()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_detections": total_detections,
                "unique_signs": unique_signs,
                "last_detection": recent_detection.timestamp.strftime('%Y-%m-%d %H:%M:%S') if recent_detection else None,
                "most_common_sign": most_common[0] if most_common else None,
                "most_common_count": most_common[1] if most_common else 0
            }
        })
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Cleanup on app shutdown
def cleanup_cameras():
    """Clean up all active cameras"""
    global active_cameras
    logger.info("🧹 Cleaning up all cameras...")
    
    for user_id, camera in list(active_cameras.items()):
        try:
            camera.stop()
            logger.info(f"✅ Stopped camera for user {user_id}")
        except Exception as e:
            logger.error(f"❌ Error stopping camera for user {user_id}: {e}")
    
    active_cameras.clear()
    logger.info("✅ All cameras cleaned up")


# Register cleanup function
atexit.register(cleanup_cameras)