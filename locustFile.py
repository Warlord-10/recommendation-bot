from locust import HttpUser, task, between

class UserApi(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_user(self):
        self.client.post("http://127.0.0.1:5000/chat", json={'prompt': "I am going to a party and I want to surprise my friend with a sweet dish I prepared for him."})