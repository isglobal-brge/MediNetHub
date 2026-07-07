from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0012_experiment_experimentjobconfig'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='privacy_delta',
            field=models.FloatField(
                blank=True,
                null=True,
                help_text='δ value used when computing ε (typically 1e-5).',
            ),
        ),
        migrations.AddField(
            model_name='trainingjob',
            name='privacy_epsilon',
            field=models.FloatField(
                blank=True,
                null=True,
                help_text='Accumulated ε reported by the Node after training (lower = stronger privacy).',
            ),
        ),
    ]
