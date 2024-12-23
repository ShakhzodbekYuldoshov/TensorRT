/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "common/bboxUtils.h"
#include <cuda_runtime_api.h>
#include "common/kernel.h"
#include "common/nmsUtils.h"
#include "gatherNMSLandmarkOutputs.h"


pluginStatus_t nmsInferenceLandmark(cudaStream_t stream,
                            const int N,
                            const int perBatchBoxesSize,
                            const int perBatchScoresSize,
                            const int perBatchLandmarksSize,
                            const bool shareLocation,
                            const int backgroundLabelId,
                            const int numPredsPerClass,
                            const int numClasses,
                            const int topK,
                            const int keepTopK,
                            const float scoreThreshold,
                            const float iouThreshold,
                            const nvinfer1::DataType DT_BBOX,
                            const void *locData,
                            const nvinfer1::DataType DT_SCORE,
                            const void *confData,
                            const void *landData,
                            void *keepCount,
                            void *nmsedBoxes,
                            void *nmsedScores,
                            void *nmsedClasses,
                            void *nmsedLandmarks,
                            void *workspace,
                            bool isNormalized,
                            bool confSigmoid,
                            bool clipBoxes,
                            int scoreBits,
                            bool caffeSemantics)
{
    // Calculate the total number of locations (batch size * number of boxes per sample * 4 coordinates per box)
    const int locCount = N * perBatchBoxesSize;

    /*
    * Determine the number of location classes:
    * If shareLocation is true, bounding boxes are shared among all classes (e.g., multi-class classification).
    * Otherwise, bounding boxes are specific to individual classes (binary classification).
    */
    const int numLocClasses = shareLocation ? 1 : numClasses;

    // Calculate the size of the bounding box data
    size_t bboxDataSize = detectionForwardBBoxDataSize(N, perBatchBoxesSize, DT_BBOX);

    // Allocate workspace for raw bounding box data
    void* bboxDataRaw = workspace;

    // Copy bounding box data from `locData` to `bboxDataRaw` on the GPU
    cudaMemcpyAsync(bboxDataRaw, locData, bboxDataSize, cudaMemcpyDeviceToDevice, stream);

    // Initialize plugin status variable
    pluginStatus_t status;

    /*
    * bboxDataRaw format:
    * [batch_size, numPriors (per sample), numLocClasses, 4 coordinates per box]
    */

    // Initialize bounding box data pointer
    void* bboxData;

    // Calculate the size required for permuted bounding box data
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, perBatchBoxesSize, DT_BBOX);

    // Allocate workspace for permuted bounding box data
    void* bboxPermute = nextWorkspacePtr(reinterpret_cast<int8_t*>(bboxDataRaw), bboxDataSize);


    /*
    * After permutation, bboxData format:
    * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
    * This is equivalent to swapping axis.
    */
    if (!shareLocation)
    {
        // Permute the bounding box data for non-shared locations
        status = permuteData(
            stream, locCount, numLocClasses, numPredsPerClass, 4, 
            DT_BBOX, /*applySigmoid=*/false, bboxDataRaw, bboxPermute);

        // Check the status and handle errors appropriately
        if (status != STATUS_SUCCESS)
        {
            return status; // Propagate the error for further handling
        }

        // Assign the permuted data to bboxData
        bboxData = bboxPermute;
    }
    else
    {
        /*
        * If shareLocation is true, numLocClasses = 1.
        * No need to permute data since it's already in linear memory format.
        */
        bboxData = bboxDataRaw;
    }

    /*
    * Conf data format:
    * [batch size, numPriors * param.numClasses, 1, 1]
    */
    const int numScores = N * perBatchScoresSize;

    // Calculate the total size of scores before NMS
    size_t totalScoresSize = detectionForwardPreNMSSize(N, perBatchScoresSize);

    // If the data type is half-precision, adjust the size accordingly
    if (DT_SCORE == nvinfer1::DataType::kHALF)
    {
        totalScoresSize /= 2; // detectionForwardPreNMSSize assumes kFLOAT
    }

    // Allocate workspace memory for scores
    void* scores = nextWorkspacePtr(reinterpret_cast<int8_t*>(bboxPermute), bboxPermuteSize);

    // Perform permutation on confidence scores
    /*
    * After permutation, scores format:
    * [batch_size, numClasses, numPredsPerClass, 1]
    */
    status = permuteData(
        stream, numScores, numClasses, numPredsPerClass, 1, 
        DT_SCORE, confSigmoid, confData, scores);

    // Check the status of permutation and handle errors
    if (status != STATUS_SUCCESS)
    {
        return status; // Propagate the error
    }

    // Calculate the size of indices needed for NMS
    size_t indicesSize = detectionForwardPreNMSSize(N, perBatchScoresSize);

    // Allocate workspace memory for indices
    void* indices = nextWorkspacePtr(reinterpret_cast<int8_t*>(scores), totalScoresSize);

    // Calculate the size of post-NMS scores
    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);

    // Adjust size if using half-precision data type
    if (DT_SCORE == nvinfer1::DataType::kHALF)
    {
        postNMSScoresSize /= 2; // detectionForwardPostNMSSize assumes kFLOAT
    }

    // Calculate the size of post-NMS indices
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK); // indices are int32

    // Allocate workspace for post-NMS scores
    void* postNMSScores = nextWorkspacePtr(reinterpret_cast<int8_t*>(indices), indicesSize);

    // Allocate workspace for post-NMS indices
    void* postNMSIndices = nextWorkspacePtr(reinterpret_cast<int8_t*>(postNMSScores), postNMSScoresSize);

    // Allocate workspace for sorting
    void* sortingWorkspace = nextWorkspacePtr(reinterpret_cast<int8_t*>(postNMSIndices), postNMSIndicesSize);

    // Handle score shift if using half-precision and scoreBits are within a specific range
    float scoreShift = 0.f;
    if (DT_SCORE == nvinfer1::DataType::kHALF && scoreBits > 0 && scoreBits <= 10)
    {
        scoreShift = 1.f;
    }

    // Sort scores per class so NMS can be applied
    status = sortScoresPerClass(
        stream, N, numClasses, numPredsPerClass, backgroundLabelId, 
        scoreThreshold, DT_SCORE, scores, indices, sortingWorkspace, 
        scoreBits, scoreShift);

    // Check for errors in sorting
    if (status != STATUS_SUCCESS)
    {
        return status; // Propagate error
    }


    // The bounding boxes are in the format [ymin, xmin, ymax, xmax].
    // FlipXY is set to true as the default implementation assumes [xmin, ymin, xmax, ymax].
    bool flipXY = true;

    // Perform Non-Maximum Suppression (NMS)
    status = allClassNMS(
        stream, N, numClasses, numPredsPerClass, topK, iouThreshold, 
        shareLocation, isNormalized, DT_SCORE, DT_BBOX, bboxData, 
        scores, indices, postNMSScores, postNMSIndices, flipXY, 
        scoreShift, caffeSemantics);

    // Check the status of NMS and handle errors
    if (status != STATUS_SUCCESS)
    {
        return status; // Propagate the error
    }

    // Sort the bounding boxes after NMS using scores
    status = sortScoresPerImage(
        stream, N, numClasses * topK, DT_SCORE, postNMSScores, 
        postNMSIndices, scores, indices, sortingWorkspace, scoreBits);

    // Check the status of sorting and handle errors
    if (status != STATUS_SUCCESS)
    {
        return status; // Propagate the error
    }

    // Gather data from the sorted bounding boxes after NMS
    status = gatherNMSLandmarkOutputs(
        stream, shareLocation, N, numPredsPerClass, numClasses, topK, 
        keepTopK, DT_BBOX, DT_SCORE, indices, scores, bboxData, 
        landData, keepCount, nmsedBoxes, nmsedScores, nmsedClasses, 
        nmsedLandmarks, clipBoxes, scoreShift);

    // Check the status of data gathering and handle errors
    if (status != STATUS_SUCCESS)
    {
        return status; // Propagate the error
    }

    // Return success status
    return STATUS_SUCCESS;

}

