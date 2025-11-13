from django import forms
from django.contrib.auth.models import User
from django.core.validators import validate_ipv46_address
from .models import UserProfile, Connection, Dataset
from django.contrib.auth.forms import PasswordChangeForm

class UserProfileForm(forms.ModelForm):
    """
    Form for updating UserProfile
    """
    class Meta:
        model = UserProfile
        fields = ['organization', 'bio']

class ConnectionForm(forms.ModelForm):
    """
    Form for creating and editing Connection objects, ensuring validation.
    Includes password handling (expects raw password, will be encrypted in view/model).
    """
    # Renombrar 'password_input' a 'password' para que coincida con el modelo.
    # El widget PasswordInput asegura que se muestre como un campo de contraseña.
    password = forms.CharField(widget=forms.PasswordInput, required=False, label="Password")
    api_key = forms.CharField(widget=forms.PasswordInput, required=False, label="API Key")

    class Meta:
        model = Connection
        # El campo 'password' del formulario sobreescribirá el del modelo aquí.
        fields = ['name', 'ip', 'port', 'username', 'password', 'api_key', 'active']
        widgets = {
            'ip': forms.TextInput(attrs={'placeholder': 'e.g., 192.168.1.100'}),
            'port': forms.NumberInput(attrs={'placeholder': 'e.g., 8000'}),
            'username': forms.TextInput(attrs={'placeholder': 'Enter username for API access'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            # Si estamos editando, hacemos los campos sensibles no requeridos y mostramos ayuda
            self.fields['password'].required = False
            self.fields['password'].help_text = "Leave blank to keep the current password."
            self.fields['api_key'].required = False
            self.fields['api_key'].help_text = "Leave blank to keep the current API key."
            self.fields['username'].required = False
        else:
            # Para nuevas conexiones, hacemos username, password y api_key obligatorios
            self.fields['username'].required = True
            self.fields['password'].required = True
            self.fields['api_key'].required = True
        # Limpiamos los valores sensibles para que no se muestren cifrados
        self.initial['password'] = ''
        self.initial['api_key'] = ''
        # Ensure IP and Port validation uses model's validators if not specified here
        self.fields['ip'].validators.append(validate_ipv46_address)
        self.fields['port'].validators.append(lambda value: forms.IntegerField(min_value=1, max_value=65535).clean(value))

# Forms for the new profile page
class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['organization', 'bio']

class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['old_password'].widget.attrs.update({'placeholder': 'Enter your current password'})
        self.fields['new_password1'].widget.attrs.update({'placeholder': 'Enter your new password'})
        self.fields['new_password2'].widget.attrs.update({'placeholder': 'Confirm your new password'}) 