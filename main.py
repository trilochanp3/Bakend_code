import datetime
from io import BytesIO
import io
from fastapi import FastAPI,Body, Form, HTTPException, UploadFile, BackgroundTasks, File  # Import File

from pydantic import BaseModel
import torch
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient  # Import directly from motor
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
import base64
import uuid
import os
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from PIL import Image  # Assuming PIL for image processing
import torchvision.transforms as transforms
from datetime import datetime
app = FastAPI()



client = AsyncIOMotorClient('mongodb://localhost:27017')
database = client['user']
collection = database['user_collection']
cllctn=database['machine_collection']
mclltn=database['machine_collecction']
nclltn=database['notification_collection']
oclltn=database['orders_collection']

class User(BaseModel):
    fullName: str
    email: str
    phoneNumber: str
    address: str
    password: str

class login(BaseModel):
    email: str
    password: str

app = FastAPI()

@app.post("/register")
async def register(user: User = Body(...)):
   
    try:
        print(f"Received user data: {user.dict()}")
        # Check for duplicate email
        # if collection.find_one({"email": user.email}):
        #     return {"message": "Email already exists"}

        # Hash password (implement a secure hashing algorithm)
        # hashed_password = hash_password(user.password)
        # user.password = hashed_password
        print(f"Received user data: {user.dict()}")
        # Insert user data into MongoDB
        collection.insert_one(user.dict())
        print(f"Received user data: {user.dict()}")
        return JSONResponse({"message": "Registration successful"}, status_code=201)

    except Exception as e:
        print(f"Error registering user: {e}")
        return JSONResponse({"message": "Registration failed"}, status_code=500)
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    

class UserLogin(BaseModel):
    email: str
    password: str
@app.post("/login")
async def login(user_credentials: UserLogin):
    print(f"Received user data: {user_credentials.dict()}")
    user = await collection.find_one({"email": user_credentials.email})
    # if user_credentials.email=="trilochan901@gmail.com":
    #     return{"message":"admin"}
    # print(f"Received user data: {user_credentials.dict()}")
    print(f"Received user data: {user}")
    if user_credentials.email=="pradhntrilochan901@gmail.com":
        if user and user_credentials.password == user["password"]:  # Direct comparison for plain text passwords
       
            return JSONResponse({
            "id": str(user["_id"]) if user.get("_id") else None,
            "name": user.get("full_name", ""),  # Handle potential missing "full_name" field
            "email": user["email"],
            "phoneNumber": user.get("phoneNumber", ""),  # Handle potential missing "phoneNumber" field
            "address": user.get("address", ""),  # Handle potential missing "address" field
            "type": "admin"
        })
    #  return {"message": "Login successful!"}
        else:
            return {"message": "No user found, please register"}
    else:
        if user and user_credentials.password == user["password"]:  # Direct comparison for plain text passwords
       
            return JSONResponse({
            "id": str(user["_id"]) if user.get("_id") else None,
            "name": user.get("full_name", ""),  # Handle potential missing "full_name" field
            "email": user["email"],
            "phoneNumber": user.get("phoneNumber", ""),  # Handle potential missing "phoneNumber" field
            "address": user.get("address", ""),  # Handle potential missing "address" field
            "type": "user"
        })
    #  return {"message": "Login successful!"}
        else:
            return {"message": "No user found, please register"}

    # except HTTPException as e:
    #     return JSONResponse({"message": str(e)}, status_code=e.status_code)
    # except Exception as e:
    #     print(f"Error logging in user: {e}")
    #     return JSONResponse({"message": "Login failed"}, status_code=500)
        


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
        

class AM(BaseModel):
    machineName :  str
    crop: str 
    userEmail : str
    userPhone : str 
    price: int  
    imageInBase64 : str 
@app.post("/add-machine")
async def add_machine(machine:AM


):
    image_storage_path="machine/"
    try:
        image_data = bytes(machine.imageInBase64, 'utf-8')  

        image_file = BytesIO(image_data)
        print(f"Received user data: {machine.dict()}")
        user = await collection.find_one({"email": machine.userEmail})
        print(user)
        if user:
            user_location = user.get("address", None)  
        else:
            user_location = None 
    except Exception as e:
        print(f"Error fetching user location: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    if not machine.imageInBase64 or not machine.machineName or not machine.crop or not machine.userEmail or not machine.price   :
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    filename = f"{machine.machineName}.jpg"
    image_path = os.path.join(image_storage_path, filename)
    with open(image_path, "wb") as f:
        f.write( image_file.read())

    user_data={
        "imageInBase64": image_path,
        "machineName":machine.machineName,
        "crop":machine.crop,
        "userEmail":machine.userEmail,
        "price":machine.price,
        "userPhone":machine.userPhone,
        "location":user_location
    }

    try:
        result= await mclltn.insert_one(user_data)
        return{"message":"machine added sucessfully"}
    except Exception as e:
        print(f'error adding details {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    
    # -------------------------------------------------------------------
    # -----------------------------------------------------------------------
    
class EP(BaseModel):
        fullName:str
        email:str
        phoneNumber:str
        address:str
        oemail:str
        
        
@app.post("/edit-profile")
async def ep(data:EP):
    print(data)
    try:
        user= await collection.find_one({"email":data.oemail})
        if user:
            result = await collection.update_one(
                {"email": data.oemail},
                {"$set": {"fullName": data.fullName, "email": data.email , "phoneNumber": data.phoneNumber, "address": data.address}},
                # {"$set": {"email": data.email}},
                # {"$set": {"phoneNumber": data.phoneNumber}},
                # {"$set": {"address": data.address}}
                )
            if result.modified_count == 1:
                return {"message": "Profile updated successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to update profile")
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f'Error updating profile: {e}')

    # -----------------------------------------------------------------
    # --------------------------------------------------------------------

    # Error:iimage not showing and i need to add a loop to send all images
class CropName(BaseModel):
    cropName: str

@app.post("/get-machines")
async def get_machines(name: CropName):
    try:
        machines = await mclltn.find({"crop": name.cropName}).to_list(length=None)
        response_data = []
        for machine in machines:
            own_email = machine.get("userEmail", None)
            real_user = await collection.find_one({"email": own_email})
            own_name = real_user.get("fullName", None) if real_user else None
            image_path = machine.get("imageInBase64", None)
            if image_path:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    encoded_image = image_data.decode("utf-8")
            else:
                encoded_image = None

            response_data.append({
                "id": str(machine["_id"]) if machine.get("_id") else None,
                "name": machine.get("machineName"),  # Handle potential missing "machineName" field
                "imageInBase64": encoded_image,
                "owner": own_name,
                "location": machine.get("location"),  # Handle potential missing "location" field
                "price": machine.get("price"),  # Handle potential missing "price" field
                "ownerContact": machine.get("userPhone"),
                
            })
            print(own_name)
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error fetching user location: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# model = torch.load("C:\\Users\\HP\\Downloads\\modell\\crop_classification_model.ptht")
# model.eval()  # Set the model to evaluation mode

# # # Define preprocessing function
# def preprocess_image(image_bytes):
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # Convert to PyTorch tensor
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize (adjust values if needed)
#     ])
#     img = Image.open(io.BytesIO(image_bytes))
#     img = transform(img)
#     return img.unsqueeze(0)  # Add a batch dimension


import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

class getcrop(BaseModel):
    imageInBase64: str
@app.post("/scan-crop")
async def scan_crop(data:getcrop):
    try:
        # print(data)
        # Convert base64 image to bytes
        new_model = models.resnet18(pretrained=True)
        num_classes = 7  # Number of classes in your dataset
        num_features = new_model.fc.in_features
        new_model.fc = nn.Linear(num_features, num_classes)
        model_path=r"C:\Users\HP\resnet_15confusionepoch_model.pth"
        new_model.load_state_dict(torch.load(model_path))
        new_model.eval()

        decoded_bytes = base64.b64decode(data.imageInBase64)
        image_file = BytesIO(decoded_bytes)
        image_path = image_file
        image = Image.open(image_path)
        preprocess = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
         output = new_model(input_batch)

        probs = torch.softmax(output, dim=1)
        confidence_score, predicted_class = torch.max(probs, 1)

        class_names = ['Rice', 'Canola', 'Cotton', 'Maize', 'Peanut', 'Sugarcane', 'Wheat']
        kclass_names=['ಅಕ್ಕಿ','ಸಾಸಿವೆ','ಹತ್ತಿ','ಜೋಳ','ಕಡಲೆಕಾಯಿ','ಕಬ್ಬು','ಗೋದಿ']
        print(confidence_score)
        if confidence_score.item() > 0.8:
            predicted_class_name = class_names[predicted_class.item()]
            kpredicted_class_name = kclass_names[predicted_class.item()]
        else:
            predicted_class_name = "No Class Found"
            kpredicted_class_name = "ಯೋಗ್ಯತೆ ಕಂಡುಬಂದಿಲ್ಲ"
        print(f'The predicted class is: {predicted_class_name}')
        new_unique_id = str(uuid.uuid4()).replace("-", "") 
        unique_id=new_unique_id.lower()
        print(unique_id)
        class_id=predicted_class.item()
        print(class_id)

        return {
            "id": class_id,
            "name": predicted_class_name,
            "knName":kpredicted_class_name
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        # -----------------------------------------------
        # model_path = "C:\\Users\\HP\\cropfull_classification_model.ptht"
        # model = models.resnet18(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, 1000)  # Corrected nn.Linear
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.eval()

        # new_model = models.resnet18(pretrained=True)
        # new_model.fc = nn.Linear(new_model.fc.in_features, 5)  # Corrected nn.Linear

        # new_model.fc.weight.data = model.fc.weight.data[:2]  # Simplified slicing
        # new_model.fc.bias.data = model.fc.bias.data[:2]  # Simplified slicing
        
        # image_data = io.BytesIO(io.b64decode(data))
        # image_path=r"C:\Users\HP\Desktop\Baceknd\machine\ricetest.jpg"
        # image = Image.open(image_path)
        # preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # ])
        # input_tensor = preprocess(image)
        # input_batch = input_tensor.unsqueeze(0)
        # with torch.no_grad():
        #     output = model(input_batch)

        # _, predicted_class = output.max(1)

        # class_names = ['Rice', 'cotton', 'maize', 'sugarcane', 'wheat','peanut','canola']
        # predicted_class_name = class_names[predicted_class.item()]

        # print(f'The predicted class is: {predicted_class_name}')
        # -------------------------------------------
#         # Preprocess the image
#         preprocessed_image = preprocess_image(image_data.getvalue())
        
#         # Perform inference
#         with torch.no_grad():
#             output = model(preprocessed_image)
#             predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class index
#             # Replace with your logic to map class index to crop name
#             predicted_crop_name = "Crop Name"  # Placeholder for actual mapping logic

#         # Generate a unique ID
       

# ------------------------------------------------------------
class cp(BaseModel):
    oldPassword:str
    newPassword:str
    email:str

@app.post("/change-password")
async def change_password(input: cp):
    try:
        user = await collection.find_one({"email": input.email})
        if user:
            # Update the password for the user with the provided email
            result = await collection.update_one(
                {"email": input.email},
                {"$set": {"password": input.newPassword}}
            )
            if result.modified_count == 1:
                return {"message": "Password updated successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to update password")
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f'Error updating password: {e}')


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

class Machine(BaseModel):
    id: str
    name: str
    imageInBase64: str
    owner: str
    location: str
    price: float
    ownerContact: str

class MachineAndUser(BaseModel):
    machine: Machine
    useremail: str

@app.post("/order-machine")      
async def receive_machine_and_user_details(data: MachineAndUser):
    try:
        # print(data)
        user= await collection.find_one({"phoneNumber":data.machine.ownerContact})
        print(user)
        if user:
            own_email=user.get("email",None)
            print(own_email)
            if own_email:
                createdAt = datetime.now() 

                await nclltn.insert_one({
                    "userEmail": data.useremail,
                    "own_email": own_email,
                    "title": "Machine Requested",
                    "machine_name": data.machine.name,
                    "body": f"user {data.useremail} has requested your machine {data.machine.name}",
                    "createdAt": createdAt,
                    "type": "action"
                })
        return{"message":"machine ordered sucessfully"}

                
    except Exception as e:
        print(f'Error updating password: {e}')


class getnoti(BaseModel):
    email:str
@app.post("/get-notifications")
async def gn(data:getnoti):
    try:
        print(data)
        notifications=[]
        async for user in nclltn.find({"own_email": data.email}):
            time = user["createdAt"].isoformat()
            notification = {
                "id": str(user["_id"]) if user.get("_id") else None,
                "title": user.get("title"),
                "body": user.get("body"),
                "createdAt": time,
                "type": user.get("type"),
                "isProcessed": "false"  # Assuming "isProcessed" is always false
            }
            print(user.get("type"))
            notifications.append(notification)
        return notifications


    except Exception as e:
        print(f'Error getting notifications: {e}')

class prsnoti(BaseModel):
    id:str
    action:str
    # uemail:str
    # oemail:str
@app.post("/process-notification")
async def pn(data:prsnoti):
    print(data)
    try:
        user= await nclltn.find_one({"_id":ObjectId(data.id)})
        print(user)
        if user:
            if data.action=="accept":
                await nclltn.insert_one({
                    "userEmail": user.get("own_email"),
                    "own_email": user.get("userEmail"),
                    "title": "Request accepted",
                    "body": f"Your  request for machine {user.get('machine_name')} is accepted",
                    "createdAt": user.get("createdAt"),
                    "type": "normal"

                })
                await oclltn.insert_one({
                    "userEmail": user.get("userEmail"),
                    "orderdate":user.get("createdAt"),
                    "machine":user.get("machine_name")
                })
                await nclltn.find_one_and_delete({"_id":ObjectId(data.id)})
            else:
                await nclltn.insert_one({
                    "userEmail": user.get("own_email"),
                    "own_email": user.get("userEmail"),
                    "title": "Request Rejected",
                    "body": f"Your  request for machine {user.get('machine_name')} is rejected",
                    "createdAt": user.get("createdAt"),
                    "type": "normal"

                })
            await nclltn.find_one_and_delete({"_id":data.id})
    except Exception as e:
        print(f'Error processing notifications: {e}')

# class MachineDetails(BaseModel):
#     id: str
#     imageInBase64: str
#     machine_asdasdasd_jpg: str
#     machineName: str
#     crop: str
#     userEmail: str
#     price: int
#     userPhone: str

class Order(BaseModel):
    id:str
    orderDate: str
    machine: str

class GetOrders(BaseModel):
    email: str

@app.post("/get-orders")
async def get_orders(data: GetOrders):
    try:
        # Assuming 'oclltn' is your MongoDB collection object
        cursor = oclltn.find({"userEmail": data.email})
        response_list = []

        async for order in cursor:
            order_date = order.get("orderdate").isoformat() if order.get("orderdate") else None
            # machine_details = MachineDetails(
            #     id="nsnksnfsknf",
            #     machineName=order.get("machine"),
            #     imageInBase64="nsnksnfsknf",
            #     owner="nsnksnfsknf",
            #     location="nsnksnfsknf",
            #     price=122,
            #     ownerContact="nsnksnfsknf"
            # )
            order_response = Order(
                id=str(order.get("_id")) if order.get("_id") else None,
                orderDate=order_date,
                machine=order.get("machine")
            )
            response_list.append(order_response)

        return response_list

    except Exception as e:
        print(f'Error getting orders: {e}')
        # Raise HTTPException with 500 status code if an error occurs
        raise HTTPException(status_code=500, detail="Error getting orders")