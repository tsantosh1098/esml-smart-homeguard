package com.example.esml;

public class DataClass {
    private String imagePath, uploadTime;

    public DataClass(){

    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }

    public String getUploadTime() {
        return uploadTime;
    }

    public void setUploadTime(String uploadTime) {
        this.uploadTime = uploadTime;
    }

    public DataClass(String imagePath, String uploadTime) {
        this.imagePath = imagePath;
        this.uploadTime = uploadTime;
    }
}