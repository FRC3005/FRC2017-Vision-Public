package com.team254.cheezdroid;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

public class SensorInterface implements SensorEventListener {
    private final SensorManager mSensorManager;
    private final Sensor mAccelerometer;
    private final Sensor mMagnetometer;
    private static SensorInterface sInst = null;
    private float[] mGravity;
    private float[] mGeomagnetic;
    private float[] mSensorAvg;
    private final int mAvgSamples;

    private SensorInterface(Context context) {
        mSensorManager = (SensorManager)context.getSystemService(context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        mSensorAvg = new float[3];
        mSensorAvg[0] = 0; //angle (magnetic)
        mSensorAvg[1] = 0; //pitch
        mSensorAvg[2] = 0; //roll
        mAvgSamples = 100;
    }

    public static SensorInterface getInstance(Context context){
        if(sInst == null)
        {
            sInst = new SensorInterface(context);
        }
        return sInst;
    }

    public static SensorInterface getInstance(){
        return sInst;
    }

    public void start() {
        //SENSOR_DELAY_FASTEST, SENSOR_DELAY_GAME,SENSOR_DELAY_NORMAL
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mMagnetometer, SensorManager.SENSOR_DELAY_FASTEST);
    }

    public void pause() {
        mSensorManager.unregisterListener(this);
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            mGravity = event.values;
        if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD)
            mGeomagnetic = event.values;
        if (mGravity != null && mGeomagnetic != null) {
            float R[] = new float[9];
            float I[] = new float[9];
            boolean success = SensorManager.getRotationMatrix(R, I, mGravity, mGeomagnetic);
            if (success) {
                float orientation[] = new float[3];
                SensorManager.getOrientation(R, orientation);

                //orientation contains: azimuth, pitch and roll
                for(int i=0;i<3;i++){
                    mSensorAvg[i] = (orientation[i] + (mSensorAvg[i] * (mAvgSamples-1)))/mAvgSamples;
                }
            }
        }
    }

    public String getSendableJsonString() {
        JSONObject j = new JSONObject();
        try {
            j.put("azimuth", mSensorAvg[0]);
            j.put("pitch", mSensorAvg[1]);
            j.put("roll", mSensorAvg[2]);
        } catch (JSONException e) {
            Log.e("SensorInterface", "Could not encode JSON");
        }
        return j.toString();
    }
}