import numpy as np

# Keypoint indices (COCO 13-keypoint subset used by YOLO-Pose)
NOSE = 0
L_SHO = 1
R_SHO = 2
L_ELB = 3
R_ELB = 4
L_WRI = 5
R_WRI = 6
L_HIP = 7
R_HIP = 8
L_KNE = 9
R_KNE = 10
L_ANK = 11
R_ANK = 12


def _is_valid(pt):
    """Check if a keypoint is detected (not zero)."""
    return not (pt[0] == 0 and pt[1] == 0)


def _dist(p1, p2):
    """Euclidean distance between two points, returning 0 if either is missing."""
    if not _is_valid(p1) or not _is_valid(p2):
        return 0.0
    return float(np.linalg.norm(p1 - p2))


def _angle_between(a, b, c):
    """
    Compute angle at vertex B formed by vectors BA and BC (in degrees).
    Returns 0.0 if any point is missing.
    
    Useful for computing elbow angle (shoulder→elbow→wrist)
    or shoulder angle (hip→shoulder→elbow).
    """
    if not _is_valid(a) or not _is_valid(b) or not _is_valid(c):
        return 0.0
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag < 1e-8:
        return 0.0
    cos_angle = np.clip(dot / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _height_diff(base, target):
    """
    Vertical height difference (Y axis is inverted in images: 0 is top).
    Returns base_y - target_y, so positive means target is ABOVE base.
    """
    if not _is_valid(base) or not _is_valid(target):
        return 0.0
    return float(base[1] - target[1])


def _velocity(curr, prev):
    """Frame-to-frame velocity (dx, dy). Returns (0,0) if either point is missing."""
    if not _is_valid(curr) or not _is_valid(prev):
        return 0.0, 0.0
    return float(curr[0] - prev[0]), float(curr[1] - prev[1])


def _angle_from_vertical(base, top):
    """
    Angle of the vector (base→top) from the vertical axis (in degrees).
    0° = perfectly vertical, 90° = horizontal.
    Used for trunk lean angle measurement.
    """
    if not _is_valid(base) or not _is_valid(top):
        return 0.0
    dx = top[0] - base[0]
    # Invert dy because Y is inverted in images (top→up is negative)
    dy = -(top[1] - base[1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return 0.0
    # Angle from positive Y axis (vertical up)
    angle = np.degrees(np.arctan2(abs(dx), dy))
    return float(angle)


def compute_kinematic_features(sequence: np.ndarray) -> np.ndarray:
    """
    Computes mathematical physics features (velocities, extensions, relative heights)
    from a sequence of raw pose keypoints.
    
    Args:
        sequence: A numpy array of shape (WINDOW_SIZE, 13, 2).
                  Values should be normalized coordinates [0, 1].
                  
    Returns:
        A numpy array of shape (WINDOW_SIZE, 13*2 + 8) containing the original
        flattened 26 coordinates plus 8 newly engineered kinematic features per frame.
    """
    window_size = sequence.shape[0]
    
    # 1. Flatten original sequence: (WINDOW_SIZE, 26)
    flat_seq = sequence.reshape(window_size, -1)
    
    # We will build an array for the new features: (WINDOW_SIZE, 8)
    new_features = np.zeros((window_size, 8), dtype=np.float32)
    
    for t in range(window_size):
        kps = sequence[t]
            
        # Feature 1 & 2: Arm extension (Shoulder to Wrist distance)
        r_ext = _dist(kps[R_SHO], kps[R_WRI])
        l_ext = _dist(kps[L_SHO], kps[L_WRI])
        
        # Feature 3 & 4: Wrist height relative to shoulder
        r_height = _height_diff(kps[R_SHO], kps[R_WRI])
        l_height = _height_diff(kps[L_SHO], kps[L_WRI])
        
        # Feature 5, 6, 7, 8: Wrist Velocity (dx/dt, dy/dt)
        if t == 0:
            r_vel_x, r_vel_y = 0.0, 0.0
            l_vel_x, l_vel_y = 0.0, 0.0
        else:
            prev_kps = sequence[t-1]
            r_vel_x, r_vel_y = _velocity(kps[R_WRI], prev_kps[R_WRI])
            l_vel_x, l_vel_y = _velocity(kps[L_WRI], prev_kps[L_WRI])
            
        new_features[t] = [
            r_ext, l_ext,            # Arm Extensions
            r_height, l_height,      # Heights above shoulder
            r_vel_x, r_vel_y,        # Right Wrist Velocity
            l_vel_x, l_vel_y         # Left Wrist Velocity
        ]
        
    # Concatenate the raw features with the engineered features
    # Output shape: (WINDOW_SIZE, 26 + 8) = (41, 34)
    enhanced_seq = np.concatenate([flat_seq, new_features], axis=1)
    
    return enhanced_seq


def compute_side_angle_features(sequence: np.ndarray) -> np.ndarray:
    """
    Computes EXPANDED biomechanical features optimized for side-angle video.
    
    Side-angle views reveal the full sagittal plane motion, allowing us to
    extract features that are invisible from a broadcast (back-of-court) angle:
    - Elbow joint angles (full extension = smash, bent = drop)
    - Shoulder joint angles (overhead = clear/smash)
    - Trunk lean angle (backward lean → forward lean = smash wind-up)
    - Wrist acceleration (2nd derivative — explosive = smash, smooth = drop)
    - Jump detection (ankle lifts off baseline)
    - Swing speed magnitude
    - Contact point height (wrist Y at max velocity)
    - Follow-through direction angle
    
    Args:
        sequence: A numpy array of shape (WINDOW_SIZE, 13, 2).
                  Values should be normalized coordinates [0, 1].
                  
    Returns:
        A numpy array of shape (WINDOW_SIZE, 26 + 16) = (WINDOW_SIZE, 42)
        containing flattened coordinates + 16 engineered features.
    """
    window_size = sequence.shape[0]
    
    # 1. Flatten original sequence: (WINDOW_SIZE, 26)
    flat_seq = sequence.reshape(window_size, -1)
    
    # 2. Build 16 engineered features per frame
    new_features = np.zeros((window_size, 16), dtype=np.float32)
    
    # Pre-compute velocities for acceleration calculation
    wrist_velocities_r = np.zeros((window_size, 2), dtype=np.float32)
    wrist_velocities_l = np.zeros((window_size, 2), dtype=np.float32)
    
    for t in range(1, window_size):
        vr_x, vr_y = _velocity(sequence[t, R_WRI], sequence[t-1, R_WRI])
        vl_x, vl_y = _velocity(sequence[t, L_WRI], sequence[t-1, L_WRI])
        wrist_velocities_r[t] = [vr_x, vr_y]
        wrist_velocities_l[t] = [vl_x, vl_y]
    
    # Baseline ankle height (first valid frame)
    baseline_r_ankle_y = 0.0
    baseline_l_ankle_y = 0.0
    for t in range(window_size):
        if _is_valid(sequence[t, R_ANK]):
            baseline_r_ankle_y = sequence[t, R_ANK, 1]
            break
    for t in range(window_size):
        if _is_valid(sequence[t, L_ANK]):
            baseline_l_ankle_y = sequence[t, L_ANK, 1]
            break
    
    for t in range(window_size):
        kps = sequence[t]
        
        # ── Features 1-2: Arm extension (shoulder→wrist distance) ──
        r_ext = _dist(kps[R_SHO], kps[R_WRI])
        l_ext = _dist(kps[L_SHO], kps[L_WRI])
        
        # ── Features 3-4: Elbow angle (shoulder→elbow→wrist) ──
        # Smash: near full extension (~160-180°)
        # Drop: typically more bent (~90-130°)
        r_elbow_angle = _angle_between(kps[R_SHO], kps[R_ELB], kps[R_WRI])
        l_elbow_angle = _angle_between(kps[L_SHO], kps[L_ELB], kps[L_WRI])
        # Normalize to [0, 1] range (0° to 180°)
        r_elbow_norm = r_elbow_angle / 180.0
        l_elbow_norm = l_elbow_angle / 180.0
        
        # ── Features 5-6: Shoulder angle (hip→shoulder→elbow) ──
        # High overhead = clear/smash (~150-180°)
        # Lower = drive/net (~60-120°)
        r_shoulder_angle = _angle_between(kps[R_HIP], kps[R_SHO], kps[R_ELB])
        l_shoulder_angle = _angle_between(kps[L_HIP], kps[L_SHO], kps[L_ELB])
        r_shoulder_norm = r_shoulder_angle / 180.0
        l_shoulder_norm = l_shoulder_angle / 180.0
        
        # ── Feature 7: Trunk lean angle ──
        # Angle from vertical of the hip-midpoint → shoulder-midpoint line
        # Smashes involve backward lean during wind-up, then forward lean
        hip_mid = (kps[L_HIP] + kps[R_HIP]) / 2.0 if (_is_valid(kps[L_HIP]) and _is_valid(kps[R_HIP])) else kps[R_HIP]
        sho_mid = (kps[L_SHO] + kps[R_SHO]) / 2.0 if (_is_valid(kps[L_SHO]) and _is_valid(kps[R_SHO])) else kps[R_SHO]
        trunk_lean = _angle_from_vertical(hip_mid, sho_mid)
        trunk_lean_norm = trunk_lean / 90.0  # Normalize assuming max 90° lean
        
        # ── Features 8-9: Wrist height relative to shoulder ──
        r_height = _height_diff(kps[R_SHO], kps[R_WRI])
        l_height = _height_diff(kps[L_SHO], kps[L_WRI])
        
        # ── Features 10-11: Wrist velocity (dx, dy) ──
        r_vel_x, r_vel_y = wrist_velocities_r[t]
        l_vel_x, l_vel_y = wrist_velocities_l[t]
        
        # ── Feature 12: Swing speed magnitude ──
        # Magnitude of dominant wrist velocity vector
        r_speed = float(np.sqrt(r_vel_x**2 + r_vel_y**2))
        l_speed = float(np.sqrt(l_vel_x**2 + l_vel_y**2))
        swing_speed = max(r_speed, l_speed)
        
        # ── Features 13-14: Wrist acceleration (2nd derivative) ──
        # Smash: explosive positive acceleration then deceleration
        # Drop: smooth, low acceleration
        if t < 2:
            r_accel = 0.0
            l_accel = 0.0
        else:
            prev_r_speed = float(np.linalg.norm(wrist_velocities_r[t-1]))
            prev_l_speed = float(np.linalg.norm(wrist_velocities_l[t-1]))
            r_accel = r_speed - prev_r_speed
            l_accel = l_speed - prev_l_speed
        
        # ── Feature 15: Jump detection ──
        # How much the ankle has risen from baseline (positive = jumping)
        # In image coords, Y decreases upward, so baseline_y - current_y > 0 means jump
        r_jump = 0.0
        l_jump = 0.0
        if _is_valid(kps[R_ANK]) and baseline_r_ankle_y > 0:
            r_jump = baseline_r_ankle_y - kps[R_ANK, 1]
        if _is_valid(kps[L_ANK]) and baseline_l_ankle_y > 0:
            l_jump = baseline_l_ankle_y - kps[L_ANK, 1]
        jump_height = max(r_jump, l_jump, 0.0)  # Clamp to non-negative
        
        # ── Feature 16: Follow-through direction angle ──
        # Angle of velocity vector from downward-vertical
        # Smash: steeply downward (~0-30°)
        # Clear: upward (~150-180°)
        # Drop: moderately downward (~30-80°)
        dominant_vel = wrist_velocities_r[t] if r_speed >= l_speed else wrist_velocities_l[t]
        if abs(dominant_vel[0]) < 1e-8 and abs(dominant_vel[1]) < 1e-8:
            follow_through = 0.0
        else:
            # Angle from positive Y direction (downward in image coords)
            follow_through = float(np.degrees(np.arctan2(abs(dominant_vel[0]), dominant_vel[1])))
            follow_through = follow_through / 180.0  # Normalize to [0, 1]
        
        new_features[t] = [
            r_ext, l_ext,                   # 1-2:   Arm extensions
            r_elbow_norm, l_elbow_norm,     # 3-4:   Elbow angles (normalized)
            r_shoulder_norm, l_shoulder_norm, # 5-6:  Shoulder angles (normalized)
            trunk_lean_norm,                # 7:     Trunk lean angle
            r_height,                       # 8:     Right wrist height above shoulder
            l_height,                       # 9:     Left wrist height above shoulder
            swing_speed,                    # 10:    Swing speed magnitude
            r_accel,                        # 11:    Right wrist acceleration
            l_accel,                        # 12:    Left wrist acceleration
            jump_height,                    # 13:    Jump detection
            follow_through,                 # 14:    Follow-through direction
            r_vel_x,                        # 15:    Right wrist velocity X
            r_vel_y,                        # 16:    Right wrist velocity Y
        ]
    
    # Concatenate: (WINDOW_SIZE, 26 + 16) = (WINDOW_SIZE, 42)
    enhanced_seq = np.concatenate([flat_seq, new_features], axis=1)
    
    return enhanced_seq
