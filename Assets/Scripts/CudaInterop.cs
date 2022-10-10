using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class CudaInterop : MonoBehaviour
{
    [DllImport("Cuda_Interop.dll", EntryPoint = "Test")]
    static extern int Test(int id);
    
    [SerializeField]
    private int send;

    IntPtr handle;


    // Start is called before the first frame update
    void Start()
    {
        int message = Test(send);
        Debug.Log(message);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
