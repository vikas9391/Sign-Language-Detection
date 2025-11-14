from flask import Blueprint, Response, render_template, jsonify, request, session
from flask_login import current_user
from datetime import datetime, timedelta
import logging
import atexit
import time
import json
from sqlalchemy import func

from detector import VideoCamera, get_latest_label, get_current_word, get_word_history
from models.detection_model import Detection
from db_setup import db

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)

# Store active camera instances per user/session
active_cameras = {}


def get_user_identifier():
    """Get user identifier - either user ID or session ID for guests"""
    if current_user.is_authenticated:
        return f"user_{current_user.id}"
    else:
        if 'guest_id' not in session:
            session['guest_id'] = f"guest_{int(time.time() * 1000)}"
        return session['guest_id']


def get_camera(identifier=None):
    """Get or create camera instance for specific user/guest"""
    global active_cameras
    
    if identifier is None:
        identifier = get_user_identifier()
    
    # Clean up old camera if exists
    if identifier in active_cameras:
        try:
            logger.info(f"🧹 Cleaning up existing camera for {identifier}")
            active_cameras[identifier].stop()
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"⚠️ Error stopping old camera for {identifier}: {e}")
    
    # Create new camera instance
    try:
        logger.info(f"📹 Initializing camera for {identifier}...")
        camera = VideoCamera()
        active_cameras[identifier] = camera
        return camera
    except Exception as e:
        logger.error(f"❌ Camera initialization error for {identifier}: {e}")
        raise


@detection_bp.route('/practice')
def practice():
    """Practice page for ASL detection - accessible to all"""
    try:
        return render_template('practice.html', user=current_user if current_user.is_authenticated else None)
    except Exception as e:
        logger.error(f"Error rendering practice page: {e}")
        return jsonify({"error": "Failed to load practice page"}), 500


@detection_bp.route('/')
@detection_bp.route('/detect')
def detect():
    """Main detection page - accessible to all"""
    return render_template('home.html')


@detection_bp.route('/video_feed')
def video_feed():
    """Video streaming route - accessible to all users"""
    try:
        identifier = get_user_identifier()
        logger.info(f"📺 Starting video feed for {identifier}")
        cam = get_camera(identifier)
        return Response(
            cam.get_frame(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"❌ Video feed error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# **NEW: Single endpoint for all detection state**
@detection_bp.route('/detection_state')
def detection_state():
    """Get complete detection state in a single request - OPTIMIZED"""
    try:
        # Get all state in one go
        label, confidence = get_latest_label()
        current_word_val = get_current_word()
        words = get_word_history()
        
        # Save detection to database only for authenticated users
        if current_user.is_authenticated and label and confidence > 0.75:
            try:
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
            except Exception as e:
                logger.error(f"❌ Error saving detection: {e}")
                db.session.rollback()
        
        # Return everything in one response
        return jsonify({
            "label": label,
            "confidence": float(confidence) if confidence else 0.0,
            "current_word": current_word_val,
            "words": words,
            "word_count": len(words)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting detection state: {e}", exc_info=True)
        return jsonify({
            "label": "",
            "confidence": 0.0,
            "current_word": "",
            "words": [],
            "word_count": 0
        }), 200


# Keep legacy endpoints for backward compatibility
@detection_bp.route('/get_label')
def get_label():
    """Get the latest detected label - DEPRECATED, use /detection_state instead"""
    try:
        label, confidence = get_latest_label()
        return jsonify({
            "label": label,
            "confidence": float(confidence) if confidence else 0.0
        }), 200
    except Exception as e:
        logger.error(f"❌ Error getting label: {e}")
        return jsonify({"label": "", "confidence": 0.0}), 200


@detection_bp.route('/get_current_word')
def get_current_word_route():
    """Get the current word - DEPRECATED, use /detection_state instead"""
    try:
        word = get_current_word()
        return jsonify({"word": word})
    except Exception as e:
        logger.error(f"Error getting current word: {e}")
        return jsonify({"word": ""})


@detection_bp.route('/get_word_history')
def get_word_history_route():
    """Get completed words - DEPRECATED, use /detection_state instead"""
    try:
        words = get_word_history()
        return jsonify({"words": words, "count": len(words)})
    except Exception as e:
        logger.error(f"Error getting word history: {e}")
        return jsonify({"words": [], "count": 0})


@detection_bp.route('/clear_word', methods=['POST'])
def clear_word():
    """Clear the current word being built"""
    try:
        identifier = get_user_identifier()
        if identifier in active_cameras:
            active_cameras[identifier].clear_current_word()
            return jsonify({"success": True, "message": "Word cleared"})
        return jsonify({"success": False, "message": "Camera not initialized"}), 400
    except Exception as e:
        logger.error(f"Error clearing word: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@detection_bp.route('/delete_letter', methods=['POST'])
def delete_letter():
    """Delete the last letter from current word"""
    try:
        identifier = get_user_identifier()
        if identifier in active_cameras:
            active_cameras[identifier].delete_last_letter()
            return jsonify({"success": True, "message": "Letter deleted"})
        return jsonify({"success": False, "message": "Camera not initialized"}), 400
    except Exception as e:
        logger.error(f"Error deleting letter: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@detection_bp.route('/stop')
@detection_bp.route('/stop_camera', methods=['GET', 'POST'])
def stop_camera():
    """Stop and release camera for current user/guest"""
    global active_cameras
    
    try:
        identifier = get_user_identifier()
        
        if identifier in active_cameras:
            try:
                logger.info(f"🛑 Stopping camera for {identifier}")
                active_cameras[identifier].stop()
                time.sleep(0.2)
                del active_cameras[identifier]
                logger.info(f"✅ Camera stopped for {identifier}")
                return jsonify({
                    "success": True,
                    "status": "Camera stopped",
                    "message": "Camera stopped"
                })
            except Exception as e:
                logger.error(f"❌ Error stopping camera for {identifier}: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        return jsonify({
            "success": True,
            "status": "Camera was not running",
            "message": "Camera already stopped"
        })
    except Exception as e:
        logger.error(f"❌ Error in stop_camera: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@detection_bp.route('/detection_status')
def detection_status():
    """Get detection system status"""
    try:
        identifier = get_user_identifier()
        is_active = identifier in active_cameras and active_cameras[identifier].is_running
        
        label, conf = get_latest_label()
        current_word_val = get_current_word()
        words = get_word_history()
        
        return jsonify({
            "active": is_active,
            "current_label": label,
            "confidence": float(conf),
            "current_word": current_word_val,
            "word_count": len(words),
            "words": words
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"active": False, "error": str(e)}), 500


# Routes below require authentication (history/stats features)
from flask_login import login_required

@detection_bp.route('/detection_history')
@login_required
def detection_history():
    """Get user's detection history - requires authentication"""
    try:
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
    """Clear user's detection history - requires authentication"""
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
    """Get user's detection statistics - requires authentication"""
    try:
        total_detections = Detection.query.filter_by(
            user_id=current_user.id
        ).count()
        
        unique_signs = db.session.query(
            Detection.detected_sign
        ).filter_by(
            user_id=current_user.id
        ).distinct().count()
        
        recent_detection = Detection.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Detection.timestamp.desc()
        ).first()
        
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


# Cleanup functions
def cleanup_cameras():
    """Clean up all active cameras"""
    global active_cameras
    logger.info("🧹 Cleaning up all cameras...")
    
    for identifier, camera in list(active_cameras.items()):
        try:
            camera.stop()
            logger.info(f"✅ Stopped camera for {identifier}")
        except Exception as e:
            logger.error(f"❌ Error stopping camera for {identifier}: {e}")
    
    active_cameras.clear()
    logger.info("✅ All cameras cleaned up")


atexit.register(cleanup_cameras)


@detection_bp.teardown_app_request
def cleanup_camera_on_error(exception=None):
    """Cleanup camera on request end if there was an error"""
    if exception:
        try:
            identifier = get_user_identifier()
            if identifier in active_cameras:
                active_cameras[identifier].stop()
                del active_cameras[identifier]
                logger.info(f"🧹 Camera cleaned up after error for {identifier}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")