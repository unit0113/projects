using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerMovement : MonoBehaviour
{
    Vector2 moveInput;
    Rigidbody2D body;
    Animator animator;
    float defaultGravityScale;
    bool isAlive = true;
    BoxCollider2D feetCollider;
    [SerializeField] float moveSpeed;
    [SerializeField] float jumpSpeed;
    [SerializeField] float climbSpeed;
    [SerializeField] Vector2 deathKick;
    [SerializeField] GameObject bullet;
    [SerializeField] Transform gun;
    
    void Start()
    {
        body = GetComponent<Rigidbody2D>();
        animator = GetComponent<Animator>();
        feetCollider = GetComponent<BoxCollider2D>();
        defaultGravityScale = body.gravityScale;
    }

    void Update()
    {
        if (!isAlive) { return; }
        Run();
        FlipSprite();
        ClimbLadder();
        Die();
    }

    void OnMove(InputValue input) {
        if (!isAlive) { return; }
        moveInput = input.Get<Vector2>();
    }

    void OnJump(InputValue input) {
        if (!isAlive) { return; }
        if (input.isPressed && feetCollider.IsTouchingLayers(LayerMask.GetMask("Ground", "Climbing"))) {
            body.velocity += new Vector2(0f, jumpSpeed);
        }
    }

    void OnFire(InputValue input) {
        if (!isAlive) { return; }
        Instantiate(bullet, gun.position, transform.rotation);
    }

    void Run() {
        body.velocity = new Vector2(moveSpeed * moveInput.x, body.velocity.y);
        animator.SetBool("isRunning", hasHorizontalVelocity());
    }

    void FlipSprite() {
        if (hasHorizontalVelocity()) {
            transform.localScale = new Vector3(Mathf.Sign(body.velocity.x), transform.localScale.y, transform.localScale.z);
        }
        
    }

    bool hasHorizontalVelocity() {
        return Mathf.Abs(body.velocity.x) > Mathf.Epsilon;
    }

    void ClimbLadder() {
        if (!feetCollider.IsTouchingLayers(LayerMask.GetMask("Climbing"))) {
            body.gravityScale = defaultGravityScale;
            animator.SetBool("isClimbing", false);
            return;
        }
        body.gravityScale = 0;
        body.velocity = new Vector2(body.velocity.x, climbSpeed * moveInput.y);
        animator.SetBool("isClimbing", Mathf.Abs(body.velocity.y) > Mathf.Epsilon);
    }

    void Die() {
        if (body.IsTouchingLayers(LayerMask.GetMask("Enemies", "Hazards"))) {
            isAlive = false;
            animator.SetTrigger("Dying");
            body.velocity = deathKick;
            FindObjectOfType<GameSession>().ProcessDeath();
        }
    }
}
