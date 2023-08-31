gcc -o pa2 pa2.s
stdbuf -oL ./pa2 <1.in >test-1.out
diff test-1.out 1.out
stdbuf -oL ./pa2 <2.in >test-2.out
diff test-2.out 2.out
stdbuf -oL ./pa2 <3.in >test-3.out
diff test-3.out 3.out
stdbuf -oL ./pa2 <4.in >test-4.out
diff test-4.out 4.out
stdbuf -oL ./pa2 <5.in >test-5.out
diff test-5.out 5.out