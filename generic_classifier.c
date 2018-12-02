#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include "iocore.h"
#include "imcore.h"
#include "cvcore.h"
#include "mlcore.h"

#define ImageWidth 128
#define ImageHeight 192
// reads all the files in train folder and train a classifier for them
// utilize the trained model on the test folder and writes the results into the results folder

// reads all the folders in the train folder and counts the number of bmp inside them
int get_groups(char *input_folder, vector_t *class_names, vector_t *file_name, vector_t *label) {

    char filename[1024];
    struct dirent *train_group, *class_member;
    // get the directory
    DIR *dr = opendir(input_folder);
    if (dr == NULL) {
        printf("cannot open directory");
        return 0;
    }
    uint32_t class_num = 0;
    // read the dir
    while (train_group = readdir (dr)) {
        // skip self and parent
        if (!strcmp(train_group->d_name, ".") || !strcmp(train_group->d_name, "..")) { continue; }
        // now we have a new class
        sprintf(filename, "%s//%s//", input_folder, train_group->d_name);
        // get the class name
        string_t cname = string(train_group->d_name);
        vector_push(class_names, &cname);
        // get the files under this class
        DIR *ct = opendir(filename);
        uint32_t member = 0;
        while (class_member = readdir(ct)) {
            if (!strcmp(class_member->d_name, ".") || !strcmp(class_member->d_name, "..")) { continue; }
            sprintf(filename, "%s//%s//%s", input_folder, train_group->d_name, class_member->d_name);
            // get the filename
            string_t fname = string(filename);
            vector_push(file_name, &fname);
            vector_push(label, &class_num);
        }
        ++class_num;
        closedir (ct);
    }
    closedir (dr);

    return length(label);
}


int main() {

    char filename[256];

    vector_t *class_name = vector_create(string_t);
    vector_t *file_name = vector_create(string_t);
    vector_t *label = vector_create(uint32_t);
    // create a feature matrix for the data
    matrix_t *img = matrix_create(uint8_t , ImageHeight, ImageWidth, 3);
    matrix_t *gray = matrix_create(uint8_t , ImageHeight, ImageWidth, 1);

    // get the train groups from the orgized folder structure
    get_groups("..//data//train", class_name, file_name, label);
    // get the number of samples and class
    uint32_t NumberOfSample = length(label);
    uint32_t NumberOfClass = length(class_name);

    // create labels
    matrix_t *labels = matrix_create(float, NumberOfSample, NumberOfClass);
    uint32_t *label_data = data(uint32_t, label);
    for(int i=0; i < NumberOfSample; i++) {
        at(float, labels, i, label_data[i]) = 1.0;
    }

    // create a feature model
    struct feature_t *hog = feature_create(CV_HOG, ImageWidth, ImageHeight, "-bins:9 -cell:8x8 -block:4x4  -stride:2x2");
    feature_view(hog);

    matrix_t *feature = matrix_create(float, NumberOfSample, feature_size(hog), 1);
    // print the class names and files under them
    for(int i=0; i < length(file_name); i++) {
        // printf("extracting features for: %s\n", at(string_t, file_name, i).data);
        // read the image
        imload(at(string_t, file_name, i)._data, img);
        rgb2gray(img, gray);
        feature_extract(gray, hog, data(float, feature, i, 0));
    }
    // we have features and labels, let classify them
    struct glm_t *net = glm_create(SVRL2,"max_iter:15000 eta:0.1 epsilon:0.4 lambda:0.1");
    glm_train(feature, label, net);
    glm_view(net);
    print_message_func(SUCCESS, __LINE__, "glm train", "training done!");

    matrix_t *train_label_predicted = matrix_create(float, NumberOfSample, NumberOfClass, 1);
    // predict the results for the test and train set
    glm_predict(feature, train_label_predicted, net);

    // create directory for the outputs
    imlab_mkdir("results");

    uint32_t train_true = 0, train_false = 0;
    uint32_t test_true = 0, test_false = 0;
    // print the results into the results folder
    for(int i=0; i < NumberOfSample; i++) {
        uint32_t maxIdx = 0;
        for(int j=0; j < NumberOfClass; j++) {
            if(at(float, train_label_predicted, i, j) > at(float, train_label_predicted, i, maxIdx)) {
                maxIdx = j;
            }
        }
        // save the image to the assigned class
        uint32_t l = at(uint32_t, label, i);
        if(l == maxIdx) {
            train_true++;
        } else {
            train_false++;
        };
    }
    printf("Training accuracy:  %3.2f\n", 100.0*train_true / (train_true+train_false));

    // start test here
    vector_t *test_class_name = vector_create(string_t);
    vector_t *test_file_name = vector_create(string_t);
    vector_t *test_label = vector_create(uint32_t);
    // create a feature matrix for the data
    matrix_t *test_img = matrix_create(uint8_t , ImageHeight, ImageWidth, 3);
    matrix_t *test_gray = matrix_create(uint8_t , ImageHeight, ImageWidth, 1);

    // get the test groups from the orgized folder structure
    get_groups("..//data//test", test_class_name, test_file_name, test_label);

    // get the number of samples and class
    uint32_t TestNumberOfSample = length(test_label);

    // extract features
    matrix_t *test_feature = matrix_create(float, TestNumberOfSample, feature_size(hog), 1);
    // print the class names and files under them
    for(int i=0; i < length(test_file_name); i++) {
        // printf("extracting features for: %s\n", at(string_t, file_name, i).data);
        // read the image
        imload(at(string_t, test_file_name, i)._data, test_img);
        rgb2gray(test_img, test_gray);
        feature_extract(test_gray, hog, data(float, test_feature, i, 0));
    }


    matrix_t *test_label_predicted = matrix_create(float, TestNumberOfSample, NumberOfClass, 1);
    // predict the results for the test and train set
    glm_predict(test_feature, test_label_predicted, net);

    // create directory for the outputs
    imlab_mkdir("results//true");
    imlab_mkdir("results//false");

    // print the results into the results folder
    for(int i=0; i < TestNumberOfSample; i++) {
        uint32_t maxIdx = 0;
        for(int j=0; j < NumberOfClass; j++) {
            if(at(float, test_label_predicted, i, j) > at(float, test_label_predicted, i, maxIdx)) {
                maxIdx = j;
            }
        }
        // save the image to the assigned class
        uint32_t l = at(uint32_t, test_label, i);
        imload(c_str(at(string_t, test_file_name, i)), img);

        if(l == maxIdx) {
            test_true++;
            sprintf(filename, "results//true//%s_%d.bmp", c_str(at(string_t, test_class_name, maxIdx)), i);
        } else {
            test_false++;
            sprintf(filename, "results//false//%s_%d.bmp", c_str(at(string_t, test_class_name, maxIdx)), i);
        };

        imwrite(img, filename);
    }
    if(TestNumberOfSample > 0) {
        printf("Testing accuracy:  %3.2f\n", 100.0 * test_true / (test_true + test_false));
    }

    vector_free(&class_name);
    vector_free(&file_name);
    //system("pause");
    return 0;
}
