using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    [SerializeField]
    private GameObject[] objects;
    private List<Camera> cameras;
    private int processID;
    
    public GameObject cameraPrefab;
    
    private void Start()
    {
        //init scene
        string[] args = System.Environment.GetCommandLineArgs();
        //set content server target
        processID = int.Parse(args[0]);
        objects[processID].SetActive(true);
        Debug.Log(processID);
        cameras = new List<Camera>();
        InitCamera();
    }

    private void InitCamera()
    {
        //TODO:number of cameras determined by java
        GameObject subCamera = GameObject.Instantiate(cameraPrefab, new Vector3(2.3f, 1.2f, 2.4f), Quaternion.Euler(20, 195, 0));
        ContentCamera cc = subCamera.GetComponent<ContentCamera>();
        cc.CameraID = 0;
        cc.ProcessID = processID;
    }
}
