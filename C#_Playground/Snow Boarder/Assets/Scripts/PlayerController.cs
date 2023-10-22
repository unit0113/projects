using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    Rigidbody2D rg2d;
    SurfaceEffector2D sE2D;
    [SerializeField] float torque_amount = 0.5f;
    [SerializeField] float boostSpeed = 30.0f;
    [SerializeField] float baseSpeed = 20.0f;
    bool canControl = true;

    // Start is called before the first frame update
    void Start()
    {
        rg2d = GetComponent<Rigidbody2D>();
        sE2D = FindObjectOfType<SurfaceEffector2D>();
    }

    // Update is called once per frame
    void Update()
    {
        if (canControl) {
            RotatePlayer();
            RespondToBoost();
        }
    }

    public void DisableControls() {
        canControl = false;
    }

    void RespondToBoost()
    {
        if (Input.GetKey(KeyCode.UpArrow)) {
            sE2D.speed = boostSpeed;
        }
        else {
            sE2D.speed = baseSpeed;
        }

    }

    void RotatePlayer()
    {
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            rg2d.AddTorque(torque_amount);
        }
        else if (Input.GetKey(KeyCode.RightArrow))
        {
            rg2d.AddTorque(-torque_amount);
        }
    }
}
