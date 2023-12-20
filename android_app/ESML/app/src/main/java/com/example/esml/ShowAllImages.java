package com.example.esml;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Build;
import android.os.Bundle;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class ShowAllImages extends AppCompatActivity {

    private RecyclerView recyclerView;
    private ArrayList<DataClass> dataList;
    private MyAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show_all_images);

        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setHasFixedSize(true);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        dataList = new ArrayList<>();
        adapter = new MyAdapter(this, dataList);
        recyclerView.setAdapter(adapter);

        Path temp_dir_path = null;
        Log.i("App started successfully", "created and running");
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            temp_dir_path = Paths.get(getFilesDir().toPath() + "/temp_dir3");
        }
        String time_stamp_file = temp_dir_path.toString() + "/timestamps.txt";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if(!Paths.get(time_stamp_file).toFile().exists()){
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

        ArrayList<MainActivity.Dataobj> data_objs = new ArrayList<>();

        for(String str: result.subList(1, result.size())){
            String[] temp = str.split("@", -2);
            data_objs.add(new MainActivity.Dataobj(temp[0], Long.parseLong(temp[1]), temp[2]));
        }
        Collections.sort(data_objs, new MainActivity.DataobjComparator());

        for(MainActivity.Dataobj data: data_objs){
            String image_path = temp_dir_path.toString() +"/"+ data.get_name();
            DataClass temp_data = new DataClass(image_path, "Captured at : " + data.get_uploaded_time());
            dataList.add(temp_data);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i("adapter len:", "len : "+ dataList.size());
        adapter.notifyItemRangeRemoved(0, adapter.getItemCount());
        recyclerView.removeAllViews();
        dataList.clear();
    }
}