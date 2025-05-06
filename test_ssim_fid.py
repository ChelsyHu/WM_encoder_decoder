{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "from pytorch_fid import fid_score\n",
    "\n",
    "\n",
    "\n",
    "path_real = 'path/to/real_images'\n",
    "path_fake = 'path/to/fake_images'\n",
    "fid_value = fid_score.calculate_fid_given_paths([path_real, path_fake], batch_size=64, device='cuda', dims=2048)\n",
    "print('FID value is:', fid_value)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
