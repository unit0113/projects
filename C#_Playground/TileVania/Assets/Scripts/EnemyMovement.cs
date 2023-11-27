using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyMovement : MonoBehaviour
{
    Rigidbody2D body;
    BoxCollider2D periscope;
    [SerializeField] float moveSpeed = 1f;
    void Start()
    {
        body = GetComponent<Rigidbody2D>();
        periscope = GetComponent<BoxCollider2D>();
    }

    void Update()
    {
        body.velocity = new Vector2(moveSpeed, 0);
    }

    void OnTriggerExit2D(Collider2D other) {
        moveSpeed = -moveSpeed;
        FlipSprite();
    }

    void FlipSprite() {
        transform.localScale = new Vector3(-transform.localScale.x, transform.localScale.y, transform.localScale.z);
    }
}
