/*
1) A person makes a call to an IVRS enabled call centre. After listening to a list of language options, he selects the preferred language. 
Further, he wishes to speak to the customer care executive. Imagine the executive is not free to take calls immediately since there are 
already 5 people waiting to place their complaint. Identify which data structure can handle this situation and perform insertion, deletion, 
and check for exception conditions, if required.

#include <stdio.h>
#include <stdlib.h>

struct Queue {
    int *language;
    int front, rear, itemCount, maxSize;
};

struct Queue* createQueue(int maxSize) {
    struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
    queue->language = (int*)malloc(maxSize * sizeof(int));
    queue->front = 0;
    queue->rear = -1;
    queue->itemCount = 0;
    queue->maxSize = maxSize;
    return queue;
}

int isFull(struct Queue* queue) {
    return queue->itemCount == queue->maxSize;
}

int isEmpty(struct Queue* queue) {
    return queue->itemCount == 0;
}

void enqueue(struct Queue* queue, int langOption) {
    if (isFull(queue)) {
        printf("The customer care executive is currently busy. Please try again later.\n");
    } else {
        queue->rear = (queue->rear + 1) % queue->maxSize;
        queue->language[queue->rear] = langOption;
        queue->itemCount++;
        printf("Your preferred language option %d is added to the queue.\n", langOption);
    }
}

void dequeue(struct Queue* queue) {
    if (isEmpty(queue)) {
        printf("The queue is empty. No more callers are waiting.\n");
    } else {
        int dequeuedLanguage = queue->language[queue->front];
        queue->front = (queue->front + 1) % queue->maxSize;
        queue->itemCount--;
        printf("Caller with preferred language option %d is being attended to.\n", dequeuedLanguage);
    }
}

void display(struct Queue* queue) {
    printf("Queue status: ");
    if (isEmpty(queue)) {
        printf("No callers are currently waiting.\n");
    } else {
        printf("Callers with language options: ");
        int i;
        for (i = 0; i < queue->itemCount; i++) {
            printf("%d ", queue->language[(queue->front + i) % queue->maxSize]);
        }
        printf("\n");
    }
}

int main() {
    int maxSize;
    printf("Enter the maximum limit of customers that can be in the queue: ");
    scanf("%d", &maxSize);

    struct Queue* queue = createQueue(maxSize);

    // Enqueue language options
    for (int i = 1; i <= maxSize + 1; i++) {
        enqueue(queue, i);
    }

    // Display queue status
    display(queue);

    // Dequeue callers
    for (int i = 0; i < maxSize; i++) {
        dequeue(queue);
    }

    return 0;
}
*/