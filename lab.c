#include <stdio.h>
#define MAX_SIZE 100 
int list[MAX_SIZE];
int listIndex = -1;
void createList (int n) 
{ if 
(n <= MAX_SIZE){
printf("Enter %d elements: \n", n);
 for(int i=0;i<n;i++){
scanf("%d", &list[i]); }
listIndex = n - 1; } else {
printf("Error: Number of elements exceeds maximum size. \n");
 }
void insertElement(int element, int position) {
if (position <= MAX _SIZE) {
for (int i = listIndex; i >= position; i--) {
list[i + 1] = list[i]; }
list[position] = element;
listIndex++; } else {
printf( 'Error: Invalid position. \n'); }
void deleteElement(int position) {
if (position >= 0 && position <= listIndex) {
for (int i = position; i < listIndex; i++) 
{ list|i] = list|i + 1];
｝
listIndex-; }
 else {
printf("Error: Invalid position. \n"); }
int searchElement(int element){
for (int i = 0; i <= listIndex; i++) { if (list[i] == element) {
