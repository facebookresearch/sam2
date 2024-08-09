# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define the URLs for the checkpoints
$baseUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
$sam2_hiera_t_url = "${baseUrl}sam2_hiera_tiny.pt"
$sam2_hiera_s_url = "${baseUrl}sam2_hiera_small.pt"
$sam2_hiera_b_plus_url = "${baseUrl}sam2_hiera_base_plus.pt"
$sam2_hiera_l_url = "${baseUrl}sam2_hiera_large.pt"

# Function to download files
function Get-File {
    param (
        [string]$url,
        [string]$filename
    )
    try {
        Invoke-WebRequest -Uri $url -OutFile $filename
        Write-Host "$filename downloaded successfully."
    }
    catch {
        Write-Host "Failed to download checkpoint from $url"
        exit 1
    }
}

# Download each of the four checkpoints
Write-Host "Downloading sam2_hiera_tiny.pt checkpoint..."
Download-File -url $sam2_hiera_t_url -filename "sam2_hiera_tiny.pt"

Write-Host "Downloading sam2_hiera_small.pt checkpoint..."
Download-File -url $sam2_hiera_s_url -filename "sam2_hiera_small.pt"

Write-Host "Downloading sam2_hiera_base_plus.pt checkpoint..."
Download-File -url $sam2_hiera_b_plus_url -filename "sam2_hiera_base_plus.pt"

Write-Host "Downloading sam2_hiera_large.pt checkpoint..."
Download-File -url $sam2_hiera_l_url -filename "sam2_hiera_large.pt"

Write-Host "All checkpoints are downloaded successfully."
