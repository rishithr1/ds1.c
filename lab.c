#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define size 25

struct stack {
    char elem[size];
    int top;
} s;

int findStringLength(char *str) {
    int l = 0;
    while (*str != '\0') {
        l++;
        str++;
    }
    return l;
}

void push(char val) {
    if (s.top == (size - 1)) {
        printf("STACK IS FULL!!\n");
    } else {
        s.top = s.top + 1;
        s.elem[s.top] = val;
    }
}

char pop() {
    if (s.top == -1) {
        printf("STACK IS EMPTY!!\n");
        return '\0';
    } else {
        char ch = s.elem[s.top];
        s.top = s.top - 1;
        return ch;
    }
}

int isEmpty() {
    return s.top == -1;
}

char peek() {
    if (s.top == -1) {
        return '\0';
    } else {
        return s.elem[s.top];
    }
}

int balance(char a[], int len) {
    int i;
    for (i = 0; i < len; i++) {
        if (a[i] == '(' || a[i] == '{' || a[i] == '[') {
            push(a[i]);
        } else if (a[i] == ')' || a[i] == '}' || a[i] == ']') {
            if (s.top == -1) {
                return i;
            } else if ((s.elem[s.top] == '(' && a[i] == ')') || (s.elem[s.top] == '[' && a[i] == ']') || (s.elem[s.top] == '{' && a[i] == '}')) {
                pop();
            } else {
                return i;
            }
        }
    }
    if (s.top == -1) {
        return -1;
    }
    return s.top;
}

int precedence(char op) {
    if (op == '+' || op == '-') {
        return 1;
    } else if (op == '*' || op == '/') {
        return 2;
    } else if (op == '^') {
        return 3;
    } else {
        return -1;
    }
}

void convertInfixToPostfix(char *infix) {
    int i;
    int j = -1;
    for (i = 0; infix[i]; ++i) {
        if (isalnum(infix[i]))
            infix[++j] = infix[i];
        else if (infix[i] == '(' || infix[i] == '[' || infix[i] == '{')
            push(infix[i]);
        else if (infix[i] == ')' || infix[i] == '}' || infix[i] == ']') {
            while (!isEmpty() && peek() != '(' && peek() != '[' && peek() != '{')
                infix[++j] = pop();
            pop();
        } else {
            while (!isEmpty() && precedence(infix[i]) <= precedence(peek()))
                infix[++j] = pop();
            push(infix[i]);
        }
    }
    while (!isEmpty())
        infix[++j] = pop();
    infix[++j] = '\0';
}

int applyOp(int a, int b, char op) {
    if (op == '+')
        return a + b;
    else if (op == '-')
        return a - b;
    else if (op == '*')
        return a * b;
    else if (op == '/') {
        if (b != 0)
            return a / b;
        else {
            printf("division by zero error!\n");
            exit(EXIT_FAILURE);
        }
    } else {
        printf("invalid operator!\n");
        exit(EXIT_FAILURE);
    }
}

int evaluatePostfix(char postfix[]) {
    int i;
    int val, A, B;
    for (i = 0; postfix[i] != '\0'; i++) {
        char ch = postfix[i];
        if (isdigit(ch)) {
            push(ch - '0');
        } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
            A = pop();
            B = pop();
            val = applyOp(B, A, ch);
            push(val);
        }
    }
    return pop();
}

int main() {
    printf("Enter the infix expression : ");
    char a[size];
    scanf("%s", a);
    int len = findStringLength(a);
    s.top = -1;
    int c = balance(a, len);
    if (c == -1) {
        printf("The expression is balanced correctly.\n");
        convertInfixToPostfix(a);
        printf("postfix expression: %s\n", a);
        int result = evaluatePostfix(a);
        printf("The result is: %d\n", result);
    } else {
        printf("expression is not balanced correctly.\n");
    }
    return 0;
}
