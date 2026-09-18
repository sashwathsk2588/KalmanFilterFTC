package org.firstinspires.ftc.teamcode;

/**
 * A lightweight, dependency-free linear Kalman filter for fusing wheel/dead-wheel
 * odometry (high-rate, relative, drifts over time) with AprilTag vision poses
 * (low-rate, absolute, noisy) into a single robot pose estimate: (x, y, heading).
 *
 * Why a linear KF is enough here (no EKF):
 *   Your odometry subsystem already integrates encoder ticks + heading into a
 *   pose in FIELD coordinates every loop. So the change it reports each loop
 *   (dx, dy, dHeading) can just be ADDED to the state -- the transition is
 *   linear (F = I). All the nonlinear trig happens inside your odometry code,
 *   not inside this filter.
 *
 * Usage each loop:
 *   1. predict(dx, dy, dHeading)  -- every loop, from odometry's pose change
 *   2. update(x, y, heading, R)   -- whenever AprilTag sees a tag this loop
 *
 * Units: use whatever length unit you want (inches or meters) as long as you're
 * consistent everywhere (state, Q, R). Angles are radians internally.
 */
public class KalmanPoseFilter {

    // ----- state: x, y, heading -----
    private double x, y, heading;

    // ----- 3x3 state covariance -----
    private final double[][] P;

    // ----- 3x3 process noise: uncertainty injected into the estimate every predict() call -----
    // Tune this to reflect how much your odometry drifts per loop. Bigger Q = filter
    // trusts AprilTag corrections more; smaller Q = filter trusts odometry more.
    private final double[][] Q;

    public KalmanPoseFilter(double startX, double startY, double startHeadingRad,
                             double[][] initialP, double[][] processNoiseQ) {
        this.x = startX;
        this.y = startY;
        this.heading = normalize(startHeadingRad);
        this.P = copy(initialP);
        this.Q = copy(processNoiseQ);
    }

    /** Convenience constructor with reasonable starting covariances. TUNE THESE for your robot. */
    public KalmanPoseFilter(double startX, double startY, double startHeadingRad) {
        this(startX, startY, startHeadingRad,
                diag(1.0, 1.0, Math.toRadians(15)),        // how unsure we are at start
                diag(0.02, 0.02, Math.toRadians(0.3)));    // odometry drift injected per loop
    }

    // ---------------------------------------------------------------------
    // PREDICT: fold in an odometry pose change (already in field coordinates)
    // ---------------------------------------------------------------------
    public synchronized void predict(double dx, double dy, double dHeadingRad) {
        x += dx;
        y += dy;
        heading = normalize(heading + dHeadingRad);

        // F = I  =>  P = P + Q
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                P[i][j] += Q[i][j];
            }
        }
    }

    // ---------------------------------------------------------------------
    // UPDATE: correct with an absolute AprilTag pose measurement.
    // R = how much to trust THIS particular measurement (see varianceForTag below).
    // ---------------------------------------------------------------------
    public synchronized void update(double measX, double measY, double measHeadingRad, double[][] R) {
        // innovation = measurement - prediction, H = I, with angle wraparound on heading
        double[] innovation = {
                measX - x,
                measY - y,
                normalize(measHeadingRad - heading)
        };

        // S = H P H^T + R = P + R   (H = I)
        double[][] S = add(P, R);
        double[][] Sinv = invert3x3(S);

        // Kalman gain: K = P H^T S^-1 = P S^-1
        double[][] K = multiply(P, Sinv);

        // state update: x = x + K * innovation
        double[] delta = multiply(K, innovation);
        x += delta[0];
        y += delta[1];
        heading = normalize(heading + delta[2]);

        // covariance update: P = (I - K) P
        double[][] IminusK = subtract(identity(), K);
        double[][] newP = multiply(IminusK, P);
        for (int i = 0; i < 3; i++) {
            System.arraycopy(newP[i], 0, P[i], 0, 3);
        }
    }

    // ---------------------------------------------------------------------
    // Heuristic: size R from tag range and detection quality. Farther / weaker
    // detections get bigger (less-trusted) variances. Tune the base constants
    // against your own camera by logging robotPose vs a known ground-truth pose.
    // ---------------------------------------------------------------------
    public static double[][] varianceForTag(double rangeMeters, double decisionMargin) {
        double posBaseVar = 0.02;              // (units)^2 at ~1m range
        double headingBaseVar = Math.toRadians(2); // rad^2-ish at ~1m range

        double rangeFactor = Math.max(1.0, rangeMeters * rangeMeters);
        double marginFactor = Math.max(1.0, 60.0 / Math.max(decisionMargin, 1.0));

        double posVar = posBaseVar * rangeFactor * marginFactor;
        double headVar = headingBaseVar * rangeFactor * marginFactor;

        return diag(posVar, posVar, headVar);
    }

    // ----- getters -----
    public synchronized double getX() { return x; }
    public synchronized double getY() { return y; }
    public synchronized double getHeading() { return heading; }

    // ---------------------------------------------------------------------
    // small matrix / angle helpers (3x3 and 3x1 only -- no external library needed)
    // ---------------------------------------------------------------------
    private static double normalize(double angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    public static double[][] diag(double a, double b, double c) {
        return new double[][]{{a, 0, 0}, {0, b, 0}, {0, 0, c}};
    }

    private static double[][] identity() {
        return diag(1, 1, 1);
    }

    private static double[][] copy(double[][] m) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++) out[i] = m[i].clone();
        return out;
    }

    private static double[][] add(double[][] a, double[][] b) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                out[i][j] = a[i][j] + b[i][j];
        return out;
    }

    private static double[][] subtract(double[][] a, double[][] b) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                out[i][j] = a[i][j] - b[i][j];
        return out;
    }

    private static double[][] multiply(double[][] a, double[][] b) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    out[i][j] += a[i][k] * b[k][j];
        return out;
    }

    private static double[] multiply(double[][] a, double[] v) {
        double[] out = new double[3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                out[i] += a[i][j] * v[j];
        return out;
    }

    /** Closed-form 3x3 inverse via the adjugate matrix -- avoids pulling in a matrix library. */
    private static double[][] invert3x3(double[][] m) {
        double a = m[0][0], b = m[0][1], c = m[0][2];
        double d = m[1][0], e = m[1][1], f = m[1][2];
        double g = m[2][0], h = m[2][1], i = m[2][2];

        double A = (e * i - f * h);
        double B = -(d * i - f * g);
        double C = (d * h - e * g);
        double D = -(b * i - c * h);
        double E = (a * i - c * g);
        double F = -(a * h - b * g);
        double G = (b * f - c * e);
        double H = -(a * f - c * d);
        double I = (a * e - b * d);

        double det = a * A + b * B + c * C;
        if (Math.abs(det) < 1e-12) {
            // Singular (shouldn't happen with sane P/R) -- fail safe rather than throw.
            return identity();
        }
        double invDet = 1.0 / det;

        return new double[][]{
                {A * invDet, D * invDet, G * invDet},
                {B * invDet, E * invDet, H * invDet},
                {C * invDet, F * invDet, I * invDet}
        };
    }
}
