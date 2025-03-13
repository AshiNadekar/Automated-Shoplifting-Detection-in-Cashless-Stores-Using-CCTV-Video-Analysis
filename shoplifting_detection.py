from inference_sdk import InferenceHTTPClient
import supervision as sv
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)
def shoplifting_classifier(img):

    result = CLIENT.infer(img, model_id="shoplifting-detection-erald/2")
    return result
def output(img):
  result=extract_person_coordinate(img)
  detections=sv.Detections.from_inference(result)
  labels=detections.class_id
  prob=detections.confidence
  print(labels,prob)
  

