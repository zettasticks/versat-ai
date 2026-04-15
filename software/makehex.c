#include <stdio.h>

#define MIN(A, B) ((A) < (B) ? (A) : (B))

long int GetFileSize(FILE *file) {
  long int mark = ftell(file);

  fseek(file, 0, SEEK_END);
  long int size = ftell(file);

  fseek(file, mark, SEEK_SET);

  return size;
}

static int CharToInt(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  } else if (ch >= 'A' && ch <= 'F') {
    return 10 + (ch - 'A');
  } else if (ch >= 'a' && ch <= 'f') {
    return 10 + (ch - 'a');
  } else {
    (*(char *)NULL) = 1;
  }
  return 0;
}

int ParseInt(char *str) {
  int res = 0;
  for (int i = 0;; i++) {
    if (str[i] == '\0') {
      break;
    }

    res *= 10;
    res += CharToInt(str[i]);
  }

  return res;
}

int main(int argc, char *argv[]) {
  char *binfile = argv[1];
  char *memsize = argv[2];
  char *outputFile = argv[3];

  int memSize = ParseInt(memsize);

  printf("In:%s\nMem:%d\nOut:%s\n", binfile, memSize, outputFile);

  FILE *in = fopen(binfile, "rb");

  char pathBuffer[1024];

  int size = sprintf(pathBuffer, "%s", binfile);
  pathBuffer[size - 4] = '_';
  pathBuffer[size - 3] = '0';
  pathBuffer[size - 2] = '.';
  pathBuffer[size - 1] = 'h';
  pathBuffer[size] = 'e';
  pathBuffer[size + 1] = 'x';

  printf("%s\n", pathBuffer);

  FILE *out0 = fopen(pathBuffer, "w");

  pathBuffer[size - 3] = '1';
  FILE *out1 = fopen(pathBuffer, "w");

  pathBuffer[size - 3] = '2';
  FILE *out2 = fopen(pathBuffer, "w");

  pathBuffer[size - 3] = '3';
  FILE *out3 = fopen(pathBuffer, "w");

  FILE *out = fopen(outputFile, "w");
  long int inSize = GetFileSize(in);

  unsigned char buffer[1024 * 4];

  for (int i = 0; i < inSize; i += 1024) {
    int leftover = MIN(inSize - i, 1024);

    fread(buffer, sizeof(unsigned char), leftover, in);

    for (int j = 0; j < leftover; j += 4) {
      fprintf(out, "%02x", buffer[j + 3]);
      fprintf(out, "%02x", buffer[j + 2]);
      fprintf(out, "%02x", buffer[j + 1]);
      fprintf(out, "%02x", buffer[j + 0]);

      fprintf(out0, "%02x\n", buffer[j + 0]);
      fprintf(out1, "%02x\n", buffer[j + 1]);
      fprintf(out2, "%02x\n", buffer[j + 2]);
      fprintf(out3, "%02x\n", buffer[j + 3]);

      fprintf(out, "\n");
    }
  }

#if 0
   long int leftover = (1 << memSize) - inSize;

   for(int i = 0; i < leftover / 4; i += 1){
      fprintf(out,"00000000\n");
   }
#endif

  fclose(in);
  fclose(out);
  fclose(out0);
  fclose(out1);
  fclose(out2);
  fclose(out3);

  return 0;
}