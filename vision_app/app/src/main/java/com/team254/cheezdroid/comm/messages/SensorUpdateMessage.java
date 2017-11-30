package com.team254.cheezdroid.comm.messages;

import com.team254.cheezdroid.SensorInterface;

public class SensorUpdateMessage extends VisionMessage {

    @Override
    public String getType() {
        return "sensor";
    }

    @Override
    public String getMessage() {
        return SensorInterface.getInstance().getSendableJsonString();
    }
}