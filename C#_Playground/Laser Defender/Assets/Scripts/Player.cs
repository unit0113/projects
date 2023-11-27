using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements.Experimental;

public class Player : MonoBehaviour {
    Vector2 moveInput;
    Vector2 minBounds;
    Vector2 maxBounds;
    [SerializeField] float moveSpeed = 20f;
    [SerializeField] float padX;
    [SerializeField] float padY;
    Shooter shooter;

    void Awake() {
        shooter = GetComponent<Shooter>();
    }

    void Start() {
        initBounds();
    }

    void initBounds() {
        Camera mainCamera = Camera.main;
        minBounds = mainCamera.ViewportToWorldPoint(new Vector2(0,0));
        maxBounds = mainCamera.ViewportToWorldPoint(new Vector2(1,1));
    }

    void Update() {
        Move();
    }

    private void Move() {
        Vector3 moveDelta = moveSpeed * moveInput * Time.deltaTime;
        Vector2 newPos = transform.position + moveDelta;
        newPos.x = Mathf.Clamp(newPos.x, minBounds.x + padX, maxBounds.x - padX);
        newPos.y = Mathf.Clamp(newPos.y, minBounds.y + padY, maxBounds.y - padY);
        transform.position = newPos;
    }

    void OnMove(InputValue input) {
        moveInput = input.Get<Vector2>();
    }

    void OnFire(InputValue input) {
        if (shooter) {
            shooter.isFiring = input.isPressed;
        }
    }
}
