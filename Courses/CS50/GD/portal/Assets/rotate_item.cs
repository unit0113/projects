using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class rotate_item : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    private Logger logger;
    // Update is called once per frame
    void Update() {
        transform.Rotate(0, 2f, 0, Space.World);
    }


}
