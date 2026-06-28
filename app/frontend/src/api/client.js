import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export const getHealth = () => api.get('/health').then(res => res.data);

export const getSamples = (limit = 50) => api.get('/samples', { params: { limit } }).then(res => res.data);

export const getSampleDetails = (sampleId) => api.get(`/sample/${sampleId}`).then(res => res.data);

export const getRandomSample = () => api.get('/random-sample').then(res => res.data);

export const runInference = (data) => api.post('/infer', data).then(res => res.data);

export const uploadCustomSample = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/upload-infer', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }).then(res => res.data);
};
