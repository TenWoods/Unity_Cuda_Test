using UnityEngine;
using System;
using System.Runtime.InteropServices;
using AOT;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

public class ContentCamera : MonoBehaviour
{
    private int _processID = -1;
    private int _cameraID = -1;
    private RenderTexture _rt;
    private Texture2D colorTexture;
    private Texture2D depthTexture;
    private RenderTexture _depthRt;
    [SerializeField]
    private Material depthMaterial;
    private Camera _camera;

    public int ProcessID
    {
        //get => _processID;
        set => _processID = value;
    }

    public int CameraID
    {
        set => _cameraID = value;
    }

    //Call C++
    [DllImport("libCuda_Interop", EntryPoint = "SendTextureIDToCuda")]
    private static extern void SendTextureIDToCuda(int texture_id, int type, int width, int height, int processID, int cameraID);
    [DllImport("libCuda_Interop", EntryPoint = "Dispose")]
    private static extern void Dispose();
    [DllImport("libCuda_Interop", EntryPoint = "GenerateNamedPipe")]
    private static extern void GenerateNamedPipe(int processID, int cameraID);
    [DllImport("libCuda_Interop")]
    private static extern IntPtr GetPostRenderFunc();

    private void Start()
    {
        _camera = GetComponent<Camera>();
        _camera.depthTextureMode = _camera.depthTextureMode | DepthTextureMode.Depth;
        //_camera.targetTexture = _rt;
        _rt = _camera.targetTexture;
        _depthRt = new RenderTexture(_rt.width, _rt.height, 32);
        depthTexture = new Texture2D(_rt.width, _rt.height);
        colorTexture = new Texture2D(_rt.width, _rt.height);
    }

    private void OnEnable() 
    {
        RenderPipelineManager.endCameraRendering += PostRender;
    }

    private void OnDisable() 
    {
        RenderPipelineManager.endCameraRendering -= PostRender;
        Dispose();
    }

    private void PostRender(ScriptableRenderContext context, Camera camera)
    {
        OnPostRender();
    }

    private void OnPostRender()
    {
        if (_cameraID == -1 || _processID == -1)
            return;
        RenderTexture.active = _rt;
        colorTexture.ReadPixels(new Rect(0, 0, _rt.width, _rt.height), 0, 0);
        colorTexture.Apply();
        Graphics.Blit(_rt, _depthRt, depthMaterial);
        depthTexture.ReadPixels(new Rect(0, 0, _depthRt.width, _depthRt.height), 0, 0);
        depthTexture.Apply();
        int colorTextureID = (int)colorTexture.GetNativeTexturePtr();
        int depthTextureID = (int)depthTexture.GetNativeTexturePtr();
        //set color texture ID
        SendTextureIDToCuda(colorTextureID, 0, colorTexture.width, colorTexture.height, _processID, _cameraID);
        //set depth texture ID
        SendTextureIDToCuda(depthTextureID, 1, depthTexture.width, depthTexture.height, _processID, _cameraID);
        //register callback for color and depth
        GL.IssuePluginEvent(GetPostRenderFunc(), colorTextureID);
        GL.IssuePluginEvent(GetPostRenderFunc(), depthTextureID);
    }
    
    //call by game manager
    public void InitPipe()
    {
        GenerateNamedPipe(_processID, _cameraID);
    }
}
