package com.example.esml;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;

import android.annotation.SuppressLint;
import java.util.Collections;
import java.util.Date;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.ListResult;
import com.google.firebase.storage.StorageMetadata;
import com.google.firebase.storage.StorageReference;

import java.io.File;
import java.io.IOException;
import java.util.Locale;


public class MainActivity extends AppCompatActivity {

    private StorageReference mStorageReference;
    private Button btn;
    private Button sync_btn;
    private long milli = 10;
    private ArrayList<String> dataList;


    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        dataList = new ArrayList<>();
        btn = (Button) findViewById(R.id.show_all_btn);
        Path temp_dir = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            try {
                Path temp_dir_path = Paths.get(getFilesDir().toPath() + "/temp_dir3");
                if (temp_dir_path.toFile().exists()){
                    temp_dir = temp_dir_path;
                }
                else{
                    temp_dir = Files.createDirectory(temp_dir_path);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        displayLatestImage();
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showAllImages();
            }
        });
        sync_btn = (Button) findViewById(R.id.sync_data);
        sync_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                syncData();
                displayLatestImage();
            }
        });
    }

    public void syncData(){

        dataList = new ArrayList<>();
        Path temp_dir = null;
        mStorageReference = FirebaseStorage.getInstance().getReference().child("images");
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            try {
                Path temp_dir_path = Paths.get(getFilesDir().toPath() + "/temp_dir3");
                if (temp_dir_path.toFile().exists()){
                    temp_dir = temp_dir_path;
                }
                else{
                    temp_dir = Files.createDirectory(temp_dir_path);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        Path finalTemp_dir = temp_dir;

        mStorageReference.listAll().addOnSuccessListener(new OnSuccessListener<ListResult>() {
            @Override
            public void onSuccess(ListResult listResult) {
                final int[] i = {0};
                for(StorageReference fileref : listResult.getItems()) {
                    try {
                        File dest = null;
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            dest = new File(finalTemp_dir.toFile(), fileref.getName());
                        }
                        File finalDest = dest;
                        fileref.getFile(dest).addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
                            @Override
                            public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {
                                Bitmap bitmap = BitmapFactory.decodeFile(finalDest.getAbsolutePath());
                                try {
                                    FileOutputStream out = new FileOutputStream(finalDest);
                                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                                    out.flush();
                                    out.close();
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            }
                        });

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    fileref.getMetadata().addOnSuccessListener(new OnSuccessListener<StorageMetadata>() {
                        @Override
                        public void onSuccess(StorageMetadata storageMetadata) {
                            milli = storageMetadata.getUpdatedTimeMillis();
                            SimpleDateFormat datetime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US);
                            Date res = new Date(milli);
                            String last_modified_time = datetime.format(res);
                            String data = storageMetadata.getName() + "@" + milli + "@" + last_modified_time;
                            dataList.add(data);
                            if (i[0]++ == listResult.getItems().size() -1){
                                Toast.makeText(MainActivity.this, "Anomalies found", Toast.LENGTH_SHORT).show();
                                Log.i("time_stamps :", String.valueOf(dataList));
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                    try {
                                        String file_name = finalTemp_dir.toFile().toString()+"/timestamps.txt";
                                        Log.i("txtfile actual path", file_name);
                                        BufferedWriter writer = new BufferedWriter(new FileWriter(file_name));
                                        for(String str: dataList){
                                            writer.write("\n"+str);
                                        }
                                        writer.close();
                                    } catch (IOException e) {
                                        throw new RuntimeException(e);
                                    }
                                }
                            }
                        }
                    });
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    Log.i("lising all files", Arrays.toString(finalTemp_dir.toFile().listFiles()));
                }
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {

            }
        });
    }

    public void displayLatestImage(){
        Path temp_dir_path = null;
        Log.i("App started successfully", "created and running");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            temp_dir_path = Paths.get(getFilesDir().toPath() + "/temp_dir3");
        }
        String time_stamp_file = temp_dir_path.toString() + "/timestamps.txt";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if(!Paths.get(time_stamp_file).toFile().exists()){

                Log.i("filepath", String.valueOf(Paths.get(time_stamp_file)));
                Log.i("No file found", "No file found for timestamp.txt");
                Toast.makeText(this, "Seems like there are no images.. Try syncing", Toast.LENGTH_LONG).show();
                return;
            }
        }
        ArrayList<String> result = new ArrayList<>();
        BufferedReader br = null;

        try{
            br = new BufferedReader(new FileReader(time_stamp_file));
            String line;
            while ((line = br.readLine()) != null){
                result.add(line);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Log.i("lising all files", Arrays.toString(temp_dir_path.toFile().listFiles()));
        }

        ArrayList<Dataobj> data_objs = new ArrayList<>();

        for(String str: result.subList(1, result.size())){
            String[] temp = str.split("@", -2);
            data_objs.add(new Dataobj(temp[0], Long.parseLong(temp[1]), temp[2]));
        }
        Collections.sort(data_objs, new DataobjComparator());
        String image_path = temp_dir_path.toString() +"/"+ data_objs.get(0).name;
        Bitmap bitmap = BitmapFactory.decodeFile(image_path);
        ((ImageView)findViewById(R.id.imageView)).setImageBitmap(bitmap);
        TextView text = (TextView) findViewById(R.id.lastModifiedTime);
        text.setText(data_objs.get(0).uploaded_time);
    }


    static class Dataobj{
        private String name;
        private long millisecs;
        private String uploaded_time;

        public Dataobj(String name, long millisecs, String uploaded_time){
            this.name = name;
            this.millisecs = millisecs;
            this.uploaded_time = uploaded_time;
        }

        public String get_uploaded_time(){
            return uploaded_time;
        }

        public String get_name(){
            return name;
        }

        public long get_millisecs(){
            return millisecs;
        }

    }

    static class DataobjComparator implements java.util.Comparator<Dataobj>{
        @Override
        public int compare(Dataobj a, Dataobj b){
            return (int) (b.get_millisecs() - a.get_millisecs());
        }
    }
    public void showAllImages(){
        Intent intent = new Intent(this, ShowAllImages.class);
        startActivity(intent);
    }
}